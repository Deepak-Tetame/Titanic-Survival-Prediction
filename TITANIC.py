import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QComboBox, QSlider, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].dropna()
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'titanic_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

model = joblib.load('titanic_model.pkl')
scaler = joblib.load('scaler.pkl')

class TitanicApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Titanic Survival Prediction")
        self.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout()
        
        self.pclass_label = QLabel("Passenger Class:")
        layout.addWidget(self.pclass_label)
        self.pclass_dropdown = QComboBox()
        self.pclass_dropdown.addItems(["1st Class", "2nd Class", "3rd Class"])
        layout.addWidget(self.pclass_dropdown)

        self.gender_label = QLabel("Gender:")
        layout.addWidget(self.gender_label)
        self.gender_dropdown = QComboBox()
        self.gender_dropdown.addItems(["Male", "Female"])
        layout.addWidget(self.gender_dropdown)

        self.age_label = QLabel("Age: 30")
        layout.addWidget(self.age_label)
        self.age_slider = QSlider(Qt.Horizontal)
        self.age_slider.setMinimum(1)
        self.age_slider.setMaximum(80)
        self.age_slider.setValue(30)
        self.age_slider.valueChanged.connect(self.update_age_label)
        layout.addWidget(self.age_slider)

        self.fare_label = QLabel("Fare: $50")
        layout.addWidget(self.fare_label)
        self.fare_slider = QSlider(Qt.Horizontal)
        self.fare_slider.setMinimum(5)
        self.fare_slider.setMaximum(500)
        self.fare_slider.setValue(50)
        self.fare_slider.valueChanged.connect(self.update_fare_label)
        layout.addWidget(self.fare_slider)

        self.embarked_label = QLabel("Embarked From:")
        layout.addWidget(self.embarked_label)
        self.embarked_dropdown = QComboBox()
        self.embarked_dropdown.addItems(["Cherbourg", "Queenstown", "Southampton"])
        layout.addWidget(self.embarked_dropdown)

        self.predict_button = QPushButton("Predict Survival")
        self.predict_button.clicked.connect(self.predict_survival)
        layout.addWidget(self.predict_button)

        self.analysis_button = QPushButton("Show Data Analysis")
        self.analysis_button.clicked.connect(self.show_analysis)
        layout.addWidget(self.analysis_button)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def update_age_label(self, value):
        self.age_label.setText(f"Age: {value}")

    def update_fare_label(self, value):
        self.fare_label.setText(f"Fare: ${value}")

    def predict_survival(self):
        try:
            pclass = self.pclass_dropdown.currentIndex() + 1
            sex = self.gender_dropdown.currentIndex()
            age = self.age_slider.value()
            fare = self.fare_slider.value()
            embarked = self.embarked_dropdown.currentIndex()

            input_data = np.array([[pclass, sex, age, fare, embarked]])
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] * 100

            if prediction == 1:
                self.result_label.setText(f"✅ Survived! ({probability:.2f}%)")
                self.result_label.setStyleSheet("color: green; font-size: 14px; font-weight: bold;")
            else:
                self.result_label.setText(f"❌ Not Survived! ({probability:.2f}%)")
                self.result_label.setStyleSheet("color: red; font-size: 14px; font-weight: bold;")

        except Exception as e:
            QMessageBox.critical(self, "Error", "Invalid input! Please enter valid values.")

    def show_analysis(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.xticks([0, 1], ["Not Survived", "Survived"], rotation=0)
        plt.title("Survival Distribution")

        plt.subplot(1, 2, 2)
        df.groupby('Pclass')['Survived'].mean().plot(kind='bar', color=['blue', 'orange', 'purple'])
        plt.xticks(rotation=0)
        plt.title("Survival Rate by Class")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TitanicApp()
    window.show()
    sys.exit(app.exec_())
