import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Step 1: Dataset (No CSV needed)
# ------------------------------------------------------------
data = pd.DataFrame({
    'Age': [22,25,30,35,40,28,32,26,45,29,23,38,33,41,36,27,39,31,24,42],
    'Gender': ['Male','Female','Male','Female','Female','Male','Female','Male','Female','Male',
               'Female','Male','Female','Male','Female','Male','Female','Male','Female','Male'],
    'Time_on_Website': [5,3,2,1,1,4,3,6,2,5,7,3,2,1,4,5,3,6,4,2],
    'Pages_Viewed': [10,5,3,2,2,7,5,12,3,10,13,4,5,2,8,9,6,11,7,3],
    'Previous_Purchase': ['Yes','No','No','No','Yes','Yes','No','Yes','No','Yes',
                          'Yes','No','No','Yes','No','Yes','Yes','No','No','Yes'],
    'Income_Level': ['High','Medium','Low','Medium','High','High','Medium','Low','Medium','High',
                     'High','Low','Low','Medium','Low','High','High','Medium','Low','Medium'],
    'Ad_Clicked': ['Yes','No','No','No','Yes','Yes','No','Yes','No','Yes',
                   'Yes','No','No','Yes','No','Yes','Yes','No','No','Yes'],
    'Device_Type': ['Mobile','Desktop','Mobile','Desktop','Desktop','Mobile','Mobile','Mobile','Desktop','Mobile',
                    'Mobile','Desktop','Desktop','Mobile','Desktop','Mobile','Mobile','Desktop','Desktop','Mobile'],
    'Purchase': ['Yes','No','No','No','Yes','Yes','No','Yes','No','Yes',
                 'Yes','No','No','Yes','No','Yes','Yes','No','No','Yes']
})

# Convert to numeric + infer object types (✅ FutureWarning fix)
data.replace({
    'Male':0, 'Female':1,
    'No':0, 'Yes':1,
    'Low':0, 'Medium':1, 'High':2,
    'Mobile':0, 'Desktop':1
}, inplace=True)
data = data.infer_objects(copy=False)

# ------------------------------------------------------------
# Step 2: Model Training
# ------------------------------------------------------------
X = data.drop('Purchase', axis=1)
y = data['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

# ------------------------------------------------------------
# Step 3: GUI Design
# ------------------------------------------------------------
window = tk.Tk()
window.title("AI Customer Purchase Predictor")
window.geometry("520x650")
window.configure(bg="#f7fcfa")

tk.Label(window, text="Customer Purchase Prediction", font=("Arial", 16, "bold"), bg="#f7fcfa", fg="#2e8b57").pack(pady=10)
tk.Label(window, text=f"Model Accuracy: {accuracy:.2f}%", bg="#f7fcfa", fg="#555", font=("Arial", 10)).pack(pady=2)

# Input creation helper
def make_input(label):
    frame = tk.Frame(window, bg="#f7fcfa")
    frame.pack(pady=4)
    tk.Label(frame, text=label, font=("Arial", 11), bg="#f7fcfa").pack(side="left", padx=8)
    e = tk.Entry(frame, width=15)
    e.pack(side="left")
    return e

age = make_input("Age:")
gender = make_input("Gender (0=Male,1=Female):")
time = make_input("Time on Website (hrs):")
pages = make_input("Pages Viewed:")
previous = make_input("Previous Purchase (1=Yes,0=No):")
income = make_input("Income Level (0=Low,1=Med,2=High):")
ad = make_input("Ad Clicked (1=Yes,0=No):")
device = make_input("Device (0=Mobile,1=Desktop):")

# ------------------------------------------------------------
# Step 4: Prediction
# ------------------------------------------------------------
def predict():
    try:
        vals = [
            float(age.get()), int(gender.get()), float(time.get()), int(pages.get()),
            int(previous.get()), int(income.get()), int(ad.get()), int(device.get())
        ]
        pred = model.predict([vals])[0]
        probs = model.predict_proba([vals])[0]
        prob_no, prob_yes = probs[0]*100, probs[1]*100

        # Plot probability graph
        plt.bar(['Not Buy','Buy'], [prob_no, prob_yes], color=['red','green'])
        plt.title('Purchase Probability')
        plt.ylabel('Probability (%)')
        plt.show()

        # Dashboard popup
        dash = Toplevel(window)
        dash.title("Prediction Dashboard")
        dash.geometry("350x300")
        dash.configure(bg="#ecf9f1")

        msg = "✅ Likely to Purchase" if pred == 1 else "❌ Not Likely to Purchase"
        tk.Label(dash, text=msg, font=("Arial", 13, "bold"), bg="#ecf9f1").pack(pady=15)
        tk.Label(dash, text=f"Probability of Purchase: {prob_yes:.2f}%", font=("Arial", 11), bg="#ecf9f1").pack()
        tk.Label(dash, text=f"Probability of Not Purchase: {prob_no:.2f}%", font=("Arial", 11), bg="#ecf9f1").pack()

        # Export
        def export_csv():
            df = pd.DataFrame([vals], columns=X.columns)
            df['Prediction'] = 'Yes' if pred == 1 else 'No'
            file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files","*.csv")])
            if file:
                df.to_csv(file, index=False)
                messagebox.showinfo("Saved", "Prediction exported successfully!")

        tk.Button(dash, text="Export to CSV", bg="#2e8b57", fg="white", command=export_csv).pack(pady=20)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers!")

# Clear function
def clear_all():
    for e in [age, gender, time, pages, previous, income, ad, device]:
        e.delete(0, tk.END)

# Buttons
tk.Button(window, text="Predict", command=predict, font=("Arial", 12, "bold"), bg="#2e8b57", fg="white", width=14).pack(pady=10)
tk.Button(window, text="Clear Fields", command=clear_all, font=("Arial", 12, "bold"), bg="#ff6f61", fg="white", width=14).pack()

tk.Label(window, text="AI-based Customer Prediction | Naive Bayes Model", bg="#f7fcfa", fg="#555", font=("Arial", 9)).pack(side="bottom", pady=10)

window.mainloop()
