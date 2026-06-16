import json
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox,
    QTabWidget, QWidget, QFormLayout, QLineEdit, QCheckBox, 
    QDoubleSpinBox, QSpinBox, QComboBox, QScrollArea, QLabel
)
from PyQt6.QtCore import Qt
from pydantic import BaseModel, ValidationError

from config.config import APP_SETTINGS, APP_CONFIG


# Відомі варіанти вибору (домени) для специфічних полів
COMBO_OPTIONS = {
    "backend": {
        "global_descriptor": ["dinov3", "dinov2"],
        "homography": ["poselib", "opencv"],
        "lightglue": ["git", "torchscript", "tensorrt"],
        "lightglue_superpoint": ["git", "torchscript", "tensorrt"],
        "lightglue_rdd": ["git", "torchscript", "tensorrt"]
    },
    "masking_strategy": ["yolo", "none"],
    "local_extractor": ["rdd", "aliked", "superpoint", "xfeat"],
    "dtype": ["float16", "float32"],
    "default_mode": ["WEB_MERCATOR", "WGS84"],
    "verify_display_mode": ["center", "center_corners", "full"],
    "verify_label_mode": ["number", "number_rmse", "full"],
    "source_type": ["file", "rtsp", "usb"],
}

class ConfigDialog(QDialog):
    """Інтерактивне діалогове вікно для редагування конфігурації (APP_SETTINGS)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Налаштування програми")
        self.resize(700, 600)
        self.field_widgets = {}  # { (group, field_name): widget }
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Створюємо вкладки на основі APP_SETTINGS
        for group_name, group_model in APP_SETTINGS:
            if isinstance(group_model, BaseModel):
                self._create_tab(group_name, group_model)

        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Скасувати зміни")
        self.btn_reset.clicked.connect(self._load_config)
        btn_layout.addWidget(self.btn_reset)

        self.btn_factory_reset = QPushButton("Скинути до стандартних")
        self.btn_factory_reset.clicked.connect(self._load_defaults)
        btn_layout.addWidget(self.btn_factory_reset)

        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Закрити")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Застосувати")
        self.btn_save.clicked.connect(self._save_config)
        self.btn_save.setDefault(True)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

    def _create_tab(self, group_name: str, model: BaseModel):
        """Створює форму для однієї групи налаштувань."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        container = QWidget()
        form_layout = QFormLayout(container)
        
        # Перебір всіх полів у Pydantic моделі
        # Support for Pydantic V2 model_fields or fallback to dict
        fields = getattr(model, "model_fields", model.__dict__)
        
        for field_name in fields:
            val = getattr(model, field_name)
            
            # Рекурсивна підтримка вкладених моделей (наприклад models.yolo)
            if isinstance(val, BaseModel):
                form_layout.addRow(QLabel(f"<b>{field_name.upper()}</b>"))
                sub_fields = getattr(val, "model_fields", val.__dict__)
                for sub_name in sub_fields:
                    sub_val = getattr(val, sub_name)
                    widget = self._create_widget(group_name, f"{field_name}.{sub_name}", sub_val)
                    form_layout.addRow(f"  {sub_name}:", widget)
                    self.field_widgets[(group_name, f"{field_name}.{sub_name}")] = widget
                form_layout.addRow(QLabel("")) # Spacer
                continue

            widget = self._create_widget(group_name, field_name, val)
            form_layout.addRow(f"{field_name}:", widget)
            self.field_widgets[(group_name, field_name)] = widget
            
        scroll.setWidget(container)
        self.tabs.addTab(scroll, group_name.replace("_", " ").title())

    def _create_widget(self, group_name: str, field_name: str, value):
        """Створює відповідний віджет (QSpinBox, QComboBox і т.д.) на основі типу значення."""
        # Перевірка на ComboBox (наперед задані варіанти)
        combo_options = None
        base_field = field_name.split('.')[-1]
        
        if base_field in COMBO_OPTIONS:
            opts = COMBO_OPTIONS[base_field]
            if isinstance(opts, dict):
                # Наприклад backend залежить від батьківської групи
                parent_prefix = field_name.split('.')[0] if '.' in field_name else group_name
                if parent_prefix in opts:
                    combo_options = opts[parent_prefix]
            else:
                combo_options = opts

        if combo_options is not None:
            cb = QComboBox()
            cb.addItems(combo_options)
            if str(value) in combo_options:
                cb.setCurrentText(str(value))
            return cb

        # Стандартні типи
        if isinstance(value, bool):
            chk = QCheckBox()
            chk.setChecked(value)
            return chk
        elif isinstance(value, int):
            sp = QSpinBox()
            sp.setRange(-999999, 999999)
            sp.setValue(value)
            return sp
        elif isinstance(value, float):
            sp = QDoubleSpinBox()
            sp.setRange(-999999.0, 999999.0)
            sp.setDecimals(8)
            sp.setValue(value)
            return sp
        elif isinstance(value, list):
            le = QLineEdit()
            import json
            le.setText(json.dumps(value))
            return le
        else:
            le = QLineEdit()
            le.setText(str(value) if value is not None else "")
            return le

    def _get_widget_value(self, widget, original_value):
        """Зчитує значення з віджета та конвертує до оригінального типу."""
        if isinstance(widget, QComboBox):
            return widget.currentText()
        elif isinstance(widget, QCheckBox):
            return widget.isChecked()
        elif isinstance(widget, QSpinBox):
            return widget.value()
        elif isinstance(widget, QDoubleSpinBox):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            text = widget.text()
            if isinstance(original_value, list):
                if not text.strip():
                    return []
                import json
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    # Якщо не валідний JSON, пробуємо просто розбити по комах
                    return [x.strip() for x in text.split(",")]
            if original_value is None and not text:
                return None
            return text
        return None

    def _load_defaults(self):
        """Скидає всі поля до заводських налаштувань (визначених у коді)."""
        from config.config import AppConfig
        default_config = AppConfig()
        
        # Оновлюємо значення в UI на основі дефолтних
        for group_name, group_model in default_config:
            if not isinstance(group_model, BaseModel):
                continue
            
            for field_name in getattr(group_model, "model_fields", group_model.__dict__):
                val = getattr(group_model, field_name)
                
                if isinstance(val, BaseModel):
                    for sub_name in getattr(val, "model_fields", val.__dict__):
                        sub_val = getattr(val, sub_name)
                        w = self.field_widgets.get((group_name, f"{field_name}.{sub_name}"))
                        if w:
                            self._set_widget_value(w, sub_val)
                else:
                    w = self.field_widgets.get((group_name, field_name))
                    if w:
                        self._set_widget_value(w, val)

    def _load_config(self):
        """Оновлює віджети з поточного APP_SETTINGS."""
        for group_name, group_model in APP_SETTINGS:
            if not isinstance(group_model, BaseModel):
                continue
            
            for field_name in getattr(group_model, "model_fields", group_model.__dict__):
                val = getattr(group_model, field_name)
                
                if isinstance(val, BaseModel):
                    for sub_name in getattr(val, "model_fields", val.__dict__):
                        sub_val = getattr(val, sub_name)
                        w = self.field_widgets.get((group_name, f"{field_name}.{sub_name}"))
                        if w:
                            self._set_widget_value(w, sub_val)
                else:
                    w = self.field_widgets.get((group_name, field_name))
                    if w:
                        self._set_widget_value(w, val)

    def _set_widget_value(self, widget, value):
        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
            widget.setValue(value)
        elif isinstance(widget, QLineEdit):
            if isinstance(value, list):
                import json
                widget.setText(json.dumps(value))
            else:
                widget.setText(str(value) if value is not None else "")

    def _save_config(self):
        """Зберігає дані з форми назад у APP_SETTINGS та APP_CONFIG."""
        try:
            # Створюємо словник оновлень
            updates = {}
            for group_name, group_model in APP_SETTINGS:
                if not isinstance(group_model, BaseModel):
                    continue
                
                group_dict = group_model.model_dump()
                
                for field_name in getattr(group_model, "model_fields", group_model.__dict__):
                    val = getattr(group_model, field_name)
                    
                    if isinstance(val, BaseModel):
                        for sub_name in getattr(val, "model_fields", val.__dict__):
                            w = self.field_widgets.get((group_name, f"{field_name}.{sub_name}"))
                            if w:
                                orig = getattr(val, sub_name)
                                group_dict[field_name][sub_name] = self._get_widget_value(w, orig)
                    else:
                        w = self.field_widgets.get((group_name, field_name))
                        if w:
                            group_dict[field_name] = self._get_widget_value(w, val)
                            
                # Валідуємо і створюємо новий об'єкт групи з усіма вкладеними моделями
                new_group_model = group_model.__class__.model_validate(group_dict)
                setattr(APP_SETTINGS, group_name, new_group_model)

            # Оновлюємо глобальний словник
            APP_CONFIG.clear()
            APP_CONFIG.update(APP_SETTINGS.model_dump())

            # Зберігаємо на диск
            from config.config import save_user_config
            save_user_config(APP_SETTINGS)
            
            QMessageBox.information(
                self, 
                "Успіх", 
                "Налаштування успішно збережено.\n\n"
                "Деякі зміни (наприклад, завантаження моделей) "
                "почнуть діяти лише після перезапуску програми або трекінгу."
            )
            self.accept()
            
        except ValidationError as e:
            QMessageBox.warning(self, "Помилка валідації", f"Неправильні параметри:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Помилка", f"Невідома помилка:\n{e}")
