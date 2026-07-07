# Математика 5-DoF Pose Graph

Довідник до `src/geometry/pose_graph/` — моделі, резидуала, якобіана й оптимізації,
що стягують афінні пози кадрів у єдину метричну систему під час пропагації калібрування.

Код: `model_5dof.py` (формули), `optimizer.py` (граф + LM/TRF),
`pruning.py`, `diagnostics.py`. Тести: `tests/test_pose_graph_*`.

---

## 1. Задача

Є кадри відео з відносними афінними перетвореннями між ними (temporal — сусідні,
spatial — loop closures) і кілька **якорів** (anchors) — кадрів із відомою метричною
позою (задані користувачем через GPS-точки). Треба знайти абсолютну афінну позу
кожного кадру, узгоджену з усіма ребрами й прибиту до якорів.

Це нелінійний метод найменших квадратів: мінімізуємо суму квадратів зважених
резидуалів ребер по вектору станів усіх вільних (не-якірних) вузлів.

---

## 2. Вектор стану вузла

Кожен кадр (вузол) описується 5-вектором (`__init__`, коментар у `optimizer.py`):

```
state = [c_x, c_y, log_sx, log_sy, θ]
```

- `c_x, c_y` — метричні координати, куди відображається **центр кадру** `(cx, cy) = (W/2, H/2)`.
- `log_sx, log_sy` — **логарифми** масштабів по осях X та Y (анізотропія: незалежні осі).
- `θ` — кут повороту (рад).

**Чому центр, а не origin-трансляція.** Зберігаючи образ центру кадру, ми
розв'язуємо трансляцію в точці, навколо якої обертання/масштаб мають нульове
плече. Це слабше корелює трансляцію з поворотом/масштабом у якобіані → кращий
кондиціонал і збіжність.

**Чому логарифми масштабів.**
1. Масштаб строго додатний; `log` знімає обмеження — оптимізатор працює у
   необмеженому просторі, без бар'єрів.
2. Композиція масштабів стає **адитивною**: `log_sx_j = log_sx_i + log_dsx`
   (те саме для ребра-резидуала й для `predict_forward`).
3. Симетрія: збільшення й зменшення масштабу вдвічі дають `±log 2` — однакові за
   модулем штрафи, чого не дає лінійний масштаб.

---

## 3. Афінна модель ↔ стан

`_affine_to_state(M, cx, cy)` (розклад через `decompose_affine_5dof`):

```
tx, ty, sx, sy, angle = decompose_affine_5dof(M)
c_x = M00*cx + M01*cy + tx          # образ центру кадру
c_y = M10*cx + M11*cy + ty
state = [c_x, c_y, log(sx), log(sy), angle]
```

`_state_to_affine(state, cx, cy, sign)` — зворотне:

```
sx, sy = clip(exp(log_sx)), clip(exp(log_sy))     # клип [1e-6, 1e6] проти вибуху
c, s   = cos θ, sin θ
M00, M01 =  c*sx,  -s*sign*sy
M10, M11 =  s*sx,   c*sign*sy
tx = c_x - (M00*cx + M01*cy)                        # відновлюємо трансляцію з центру
ty = c_y - (M10*cx + M11*cy)
```

**Sign convention.** `sign ∈ {+1, −1}`. У `fix_node` при `det(M) < 0` для якоря
ставиться `self._sign = -1`. Це віддзеркалення (reflection) — коли метрична
система має протилежну орієнтацію осі Y (типово: піксельні координати «вниз»
проти світових «вгору»). `sign` множить Y-колонку матриці й входить у прогноз та
кутовий резидуал (див. нижче), тримаючи всю модель у єдиній хіральності.

---

## 4. Параметризація ребра

`add_edge` розкладає відносний афін `M` (from → to) і зберігає у `GraphEdge`:

```
dtx, dty       — зсув центру: (M·center − center) по X, Y
log_dsx, log_dsy — log-відношення масштабів
dtheta         — відносний кут
weight         — вага ребра (temporal < spatial < ...); anchor-ребра важать 1e6
edge_type      — "temporal" | "spatial"
```

---

## 5. Прогноз пози (predict forward/inverse)

Спільне ядро — `_predicted_translation` (`model_5dof.py`), єдине джерело кроку
«повернути-масштабувати дельту й додати до центру батька»:

```
pred_tx = tx_i + cosθ_i · sx_i · dtx − sign · sinθ_i · sy_i · dty
pred_ty = ty_i + sinθ_i · sx_i · dtx + sign · cosθ_i · sy_i · dty
```

`_predict_forward(state_i, edge, sign)` (BFS-ініціалізація сусіда з вузла i):

```
[pred_tx, pred_ty, log_sx_i + log_dsx, log_sy_i + log_dsy, θ_i + sign·dtheta]
```

`_predict_inverse` — те саме в інший бік (інвертує ребро: `1/ds`, `−dtheta`,
повернутий назад зсув), потім застосовує той самий `_predicted_translation`.

---

## 6. Резидуал ребра

`edge_residual` (`model_5dof.py`) — ЄДИНЕ місце формули; його викликають і
векторний шлях оптимізатора (`_residuals_vec`), і пер-ребровий діагностичний
(`_single_edge_residual`). Для ребра i→j (5 компонент):

```
r0 = (w / sx_i) · (c_x_j − pred_tx)
r1 = (w / sy_i) · (c_y_j − pred_ty)
r2 = (w · cx) · (log_sx_j − log_sx_i − log_dsx)
r3 = (w · cx) · (log_sy_j − log_sy_i − log_dsy)
r4 = (w · cx) · atan2( sin Δθ, cos Δθ ),   Δθ = θ_j − θ_i − sign·dtheta
```

де `cx = W/2` (пів-ширина кадру), `w = edge.weight`.

**Ваги — головна ідея.** Компоненти мають різні одиниці; ваги зводять їх до
спільної «піксельної» шкали, щоб один LM-крок бачив їх співмірними:

- **Трансляція `r0, r1`**: ділення на `sx_i`/`sy_i` нормалізує метричну похибку
  центру відносно локального масштабу кадру — похибка в «одиницях кадру», а не в
  метрах, тож кадри різного GSD зважені однаково.
- **Масштаб/кут `r2, r3, r4`**: множник `cx` — це **плече** (moment arm). Зсув
  `Δlog s` зміщує точку на відстані `cx` від центру приблизно на `cx·Δlog s`
  пікселів; поворот `Δθ` — приблизно на `cx·Δθ`. Тобто `w·cx` перетворює
  безрозмірні log-масштаб і радіани на піксель-еквівалент трансляції.
- **`atan2(sin, cos)`** обгортає кутову різницю у `(−π, π]` — коректна метрика на
  колі (без розриву на ±π).

---

## 7. Регуляризатор ізотропії

Крім ребрових, на КОЖЕН вільний вузол додається резидуал (`_residuals_vec`):

```
r_reg = 200 · cx · (log_sx − log_sy)
```

5-DoF модель дозволяє незалежні `sx ≠ sy`, але фізично масштаб дрон-камери майже
ізотропний. Цей доданок м'яко стягує `sx ≈ sy` з великою вагою `200·cx`,
не даючи зайвому ступеню свободи вироджувати граф у розтяг, але все ще
допускаючи справжню анізотропію, коли дані її сильно підтримують.

---

## 8. Аналітичний якобіан

`_jacobian_vec` (`optimizer.py`) будує розріджену матрицю ∂r/∂x. Нижче — вивід
блоків (позначення коду в дужках); звірено з 2-point FD у
`tests/test_pose_graph_jacobian.py`. Скорочення: `inv_sx = 1/sx_i`,
`syx = sy_i/sx_i = e^{ly_i−lx_i}`, `sxy = sx_i/sy_i`, `c = cosθ_i`, `s = sinθ_i`.

### Похідні r0 (по вузлу i)

Оскільки `r0 = w·inv_sx·(c_x_j − pred_tx)`, а від `log_sx_i` залежать і `inv_sx`,
і `pred_tx`:

```
∂r0/∂c_x_i = −w·inv_sx                              (j0_txi)
∂r0/∂log_sx_i = −w·c·dtx − r0                       (j0_lxi = −w·ci·dtx − res0)
∂r0/∂log_sy_i =  w·sign·s·dty·syx                   (j0_lyi)
∂r0/∂θ_i     =  w·s·dtx + w·sign·c·dty·syx          (j0_thi)
```

Вивід `∂r0/∂log_sx_i`: `d(inv_sx)/dlx_i = −inv_sx`, `d(pred_tx)/dlx_i = c·sx_i·dtx`,
тож `∂r0/∂lx_i = −inv_sx·w·(c_x_j−pred_tx) − w·inv_sx·c·sx_i·dtx = −r0 − w·c·dtx`. ✔

### Похідні r1 (по вузлу i)

```
∂r1/∂c_y_i = −w·inv_sy                              (j1_tyi)
∂r1/∂log_sx_i = −w·s·dtx·sxy                        (j1_lxi)
∂r1/∂log_sy_i = −w·sign·c·dty − r1                  (j1_lyi = −w·sign·ci·dty − res1)
∂r1/∂θ_i     = −w·c·dtx·sxy + w·sign·s·dty          (j1_thi)
```

### Похідні по вузлу j (to)

```
∂r0/∂c_x_j = w·inv_sx                               (j0_txj)
∂r1/∂c_y_j = w·inv_sy                               (j1_tyj)
∂r2/∂log_sx_j =  w·cx,  ∂r2/∂log_sx_i = −w·cx       (±jcx)
∂r3/∂log_sy_j =  w·cx,  ∂r3/∂log_sy_i = −w·cx       (±jcx)
∂r4/∂θ_j     =  w·cx,  ∂r4/∂θ_i     = −w·cx        (±jcx)
```

r2/r3/r4 лінійні по log-масштабах і куту, тож їхні похідні — константи `±w·cx`
(похідна обгорнутого `atan2(sin Δθ, cos Δθ)` по θ дорівнює ±1 усюди, крім міри-нуль).

### Регуляризатор

```
∂r_reg/∂log_sx = 200·cx,   ∂r_reg/∂log_sy = −200·cx
```

Фіксовані вузли (якорі) не є змінними — вони входять у резидуали як константні
стани (`X_full`), тому мають нульові стовпці якобіана.

---

## 9. Оптимізація

`optimize()` (`optimizer.py`):

1. **Ініціалізація x0.** `initialize_from_bfs()` — BFS від якорів, поширення станів
   через `predict_forward/inverse` по ребрах. `warm_start_from_affines()` — сід із
   попереднього розв'язку (швидша ітерація, коли користувач посунув один якір).
2. **Розв'язок.** `scipy.optimize.least_squares(method="trf")` (Trust Region
   Reflective) з розрідженим якобіаном — аналітичним (`use_analytic_jac`) або
   `2-point` FD із заданим патерном розрідженості. `max_nfev = max_iter · n_vars`.
3. **Two-stage prune** (`two_stage_prune`, `pruning.py`): після першого L2
   викидаємо spatial-викиди за MAD-порогом `median + k·1.4826·MAD` (усередині класу
   spatial, ≤ `max_frac` за прохід, ніколи не роз'єднуючи вузол від якорів) і
   пере-розв'язуємо з теплим стартом.

---

## 10. Чому чистий L2 (без robust loss)

Свідоме рішення (коментар у `config.py` / `GraphOptimizationConfig`): spatial-ребра
(loop closures) мають природно великі резидуали, і robust loss (`soft_l1`)
пригнічував би саме їх — якорі тримали б свої кадри, а решта «пливла». Чистий L2
дає loop closures повний вплив і стягує граф у форму. Викиди відсікаються НЕ
robust-функцією, а явними механізмами: edge gating (фізичні гейти до оптимізації)
та two-stage MAD-prune (див. п.9).

---

## Посилання

- Формули: `src/geometry/pose_graph/model_5dof.py`
  (`edge_residual`, `_predicted_translation`, `_affine_to_state`, `_state_to_affine`,
  `_predict_forward`, `_predict_inverse`).
- Оптимізатор і якобіан: `src/geometry/pose_graph/optimizer.py`
  (`optimize`, `_residuals_vec`, `_jacobian_vec`).
- Prune/діагностика: `pruning.py`, `diagnostics.py`.
- Конфіг: `config/graph.py` (`GraphOptimizationConfig`, `ProjectionConfig`).
- Тести: `test_pose_graph_jacobian` (аналітичний J vs FD), `_regression`,
  `_two_stage`, `_warmstart`, `_diagnostics`, `_realistic`, `test_optimizer_contract`.
