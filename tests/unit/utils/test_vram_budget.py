"""VramBudget — політика LRU-виселення моделей, витягнута з ModelManager.

До винесення ця логіка була неперевірною: вона сиділа всередині класу, що
імпортує torch і вантажить реальні ваги. Тут вільна VRAM і саме вивантаження —
інжектовані колбеки, тож перевіряється саме ПОЛІТИКА: кого виселяти, скільки
разів, і що робити, коли все закріплене.
"""

import time

from src.models.vram import VramBudget


class _Gpu:
    """Фейкова VRAM: кожне вивантаження повертає ``per_model_mb`` мегабайт."""

    def __init__(self, free_mb=1000.0, per_model_mb=1000.0):
        self.free = float(free_mb)
        self.per_model_mb = float(per_model_mb)
        self.loaded: dict[str, object] = {}
        self.unloaded: list[str] = []

    def load(self, name):
        self.loaded[name] = object()

    def free_mb(self):
        return self.free

    def unload(self, name):
        # Вірно відтворює ModelManager._unload_model_unsafe: якщо моделі немає
        # серед завантажених, виклик нічого не звільняє.
        self.unloaded.append(name)
        if self.loaded.pop(name, None) is not None:
            self.free += self.per_model_mb


def _budget(gpu, **kw):
    return VramBudget(free_vram_mb=gpu.free_mb, unload=gpu.unload, **kw)


def _touch_in_order(b, names):
    for n in names:
        b.touch(n)
        time.sleep(0.001)  # мітки — time.time(), розводимо їх у часі


class TestEvictionOrder:
    def test_evicts_least_recently_used_first(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        for n in ("yolo", "dino", "aliked"):
            gpu.load(n)
        b = _budget(gpu)
        _touch_in_order(b, ["yolo", "dino", "aliked"])  # yolo найдавніший

        evicted = b.ensure(required_mb=1000.0, loaded=gpu.loaded)

        assert evicted == ["yolo"]
        assert gpu.unloaded == ["yolo"]

    def test_evicts_repeatedly_until_budget_is_met(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=500.0)
        for n in ("a", "b", "c"):
            gpu.load(n)
        bud = _budget(gpu)
        _touch_in_order(bud, ["a", "b", "c"])

        evicted = bud.ensure(required_mb=1000.0, loaded=gpu.loaded)

        assert evicted == ["a", "b"]  # 2 × 500 МБ = рівно бюджет
        assert "c" in gpu.loaded

    def test_no_eviction_when_budget_already_met(self):
        gpu = _Gpu(free_mb=4000.0)
        gpu.load("dino")
        bud = _budget(gpu)
        bud.touch("dino")

        assert bud.ensure(required_mb=1000.0, loaded=gpu.loaded) == []
        assert gpu.unloaded == []


class TestPinning:
    def test_pinned_model_is_never_evicted(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        for n in ("yolo", "dino"):
            gpu.load(n)
        bud = _budget(gpu)
        _touch_in_order(bud, ["yolo", "dino"])
        bud.pin(["yolo"])  # найдавніший, але закріплений

        evicted = bud.ensure(required_mb=1000.0, loaded=gpu.loaded)

        assert evicted == ["dino"]
        assert "yolo" in gpu.loaded

    def test_all_pinned_gives_up_instead_of_looping(self):
        # Головний контракт: закріплення сильніше за бюджет. Без цього виходу
        # цикл крутився б вічно, бо вільна пам'ять ніколи не зросте.
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        for n in ("yolo", "dino"):
            gpu.load(n)
        bud = _budget(gpu)
        _touch_in_order(bud, ["yolo", "dino"])
        bud.pin(["yolo", "dino"])

        assert bud.ensure(required_mb=1000.0, loaded=gpu.loaded) == []
        assert gpu.unloaded == []

    def test_unpin_all_restores_eviction(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        gpu.load("yolo")
        bud = _budget(gpu)
        bud.touch("yolo")
        bud.pin(["yolo"])
        bud.unpin_all()

        assert bud.ensure(required_mb=1000.0, loaded=gpu.loaded) == ["yolo"]


class TestGuards:
    def test_empty_loaded_dict_terminates(self):
        # Пам'ять зайняв хтось інший (інший процес): виселяти нема кого, і цикл
        # мусить завершитись, а не крутитись на постійно малій вільній VRAM.
        gpu = _Gpu(free_mb=0.0)
        bud = _budget(gpu)
        assert bud.ensure(required_mb=99999.0, loaded={}) == []

    def test_disabled_budget_is_a_noop_on_cpu(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        gpu.load("dino")
        bud = _budget(gpu, enabled=False)
        bud.touch("dino")

        assert bud.ensure(required_mb=99999.0, loaded=gpu.loaded) == []
        assert gpu.unloaded == []

    def test_default_required_is_used_when_not_given(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=100.0)
        gpu.load("a")
        bud = _budget(gpu, default_required_mb=50.0)
        bud.touch("a")

        assert bud.ensure(loaded=gpu.loaded) == ["a"]

    def test_stale_usage_entry_cannot_loop_forever(self):
        # Регресія: найдавніша LRU-мітка вказує на модель, якої вже немає серед
        # завантажених, тож unload нічого не звільняє. Без зняття мітки цикл
        # обирав би те саме ім'я нескінченно (так було й до винесення класу).
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        gpu.load("real")
        bud = _budget(gpu)
        _touch_in_order(bud, ["stale", "real"])  # stale найдавніший, не завантажений

        evicted = bud.ensure(required_mb=1000.0, loaded=gpu.loaded)

        assert evicted == ["stale", "real"]  # прогрес є, цикл завершився
        assert bud.usage == {}

    def test_forget_removes_lru_entry(self):
        gpu = _Gpu(free_mb=0.0, per_model_mb=1000.0)
        gpu.load("a")
        gpu.load("b")
        bud = _budget(gpu)
        _touch_in_order(bud, ["a", "b"])
        bud.forget("a")  # модель уже вивантажена іншим шляхом

        assert bud.ensure(required_mb=1000.0, loaded=gpu.loaded) == ["b"]
