"""Application-wide Qt styling."""

APP_STYLESHEET = """
QWidget {
    background: #111820;
    color: #e6edf3;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QLabel {
    background: transparent;
}
QMainWindow, QStackedWidget {
    background: #111820;
}
QFrame#header, QFrame#sidebar, QFrame#metricsPanel {
    background: #151e27;
    border: 1px solid #263442;
}
QLabel#appTitle {
    font-size: 18pt;
    font-weight: 700;
}
QLabel#pageTitle {
    font-size: 17pt;
    font-weight: 650;
}
QLabel#sectionTitle {
    font-size: 11pt;
    font-weight: 650;
}
QLabel#muted, QLabel#metricLabel {
    color: #93a4b5;
}
QLabel#metricValue {
    color: #f4f7fa;
    font-size: 11pt;
    font-weight: 650;
}
QLabel#warning {
    background: #33270f;
    border: 1px solid #795d18;
    border-radius: 6px;
    color: #ffd978;
    padding: 8px;
}
QListWidget#navigation {
    background: transparent;
    border: 0;
    outline: 0;
    padding: 6px;
}
QListWidget#navigation::item {
    border-radius: 6px;
    margin: 2px 0;
    padding: 11px 12px;
}
QListWidget#navigation::item:selected {
    background: #245fa8;
    color: white;
}
QListWidget#navigation::item:hover:!selected {
    background: #1d2a36;
}
QPushButton {
    background: #223140;
    border: 1px solid #38506a;
    border-radius: 5px;
    padding: 7px 12px;
}
QPushButton:hover {
    background: #2a3d50;
}
QPushButton:pressed {
    background: #1b2835;
}
QPushButton:disabled {
    color: #667684;
    background: #18222c;
    border-color: #263442;
}
QPushButton#primary {
    background: #246fc2;
    border-color: #3989dd;
    color: white;
    font-weight: 600;
}
QPushButton#danger {
    color: #ffb2ad;
    border-color: #7d3b38;
}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QListWidget {
    background: #0d141b;
    border: 1px solid #304153;
    border-radius: 5px;
    padding: 6px;
    selection-background-color: #246fc2;
}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
    border-color: #4c98e8;
}
QProgressBar {
    background: #202c38;
    border: 1px solid #304153;
    border-radius: 7px;
    color: #f4f7fa;
    font-weight: 600;
    min-height: 16px;
    max-height: 16px;
    text-align: center;
}
QProgressBar::chunk {
    background: #3c8dde;
    border-radius: 6px;
}
QProgressBar#workflowProgress {
    font-size: 10pt;
    min-height: 28px;
    max-height: 28px;
}
QStatusBar {
    background: #151e27;
    border-top: 1px solid #263442;
}
QToolTip {
    color: #e6edf3;
    background: #1b2732;
    border: 1px solid #3a5066;
}
"""
