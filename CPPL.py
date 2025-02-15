import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk

from kmean_tt import *

data_df = pd.read_csv("C:\hoctap\CodeML\HetMon\patient_priority.csv")
# data_df = data_df.drop("id", axis=1)
le = preprocessing.LabelEncoder()
dt = data_df.apply(le.fit_transform)
data_df = np.array(dt[["age","gender","chest pain type","blood pressure","cholesterol","max heart rate","exercise angina","plasma glucose","skin_thickness","insulin","bmi",'diabetes_pedigree','hypertension','heart_disease','Residence_type']].values)
data_Train, data_Test = train_test_split(data_df, test_size=0.1, shuffle=True)

window = Tk()
#n_init: so lan chay thuat toan k_means voi cac trung tam khac nhau
Font_title = 'Times New Roman',20,'bold'
Font_text = 'Times New Roman',14
Text_color = '#fa0606'
Bg_Color = '#E2F7B5'

def setup_window():
    window.title("Phân cụm mức đô ưu tiên bệnh nhân")
    window.geometry("950x800+100+100")
    window["background"] = "#E2F7B5"
    window.iconbitmap("C:\hoctap\CodeML\BTN_3\heart.ico")
    window.resizable(False,False)
def score():
    lst = []
    centroids = []
    max = -99999
    n = None
    dav = None
    centroids_best = None
    for i in range(2,11):
        km = kmeans(data_Train,i)
        label = predict_cluster(data_Test, km[-1])
        lst.append((i, silhouette_score(data_Test, label), davies_bouldin_score(data_Test, label)))
        centroids.append((i, km[-1]))
    for i in lst:
        if(i[1] > max):
            max = i[1]
            n = i[0]
            dav = i[2]
    for i in centroids:
        if(i[0] == n):
            centroids_best = i[1]
    return lst, n, max, dav, centroids_best

lst, n_clusters, max_score,davies ,centroids_best = score()

lst.insert(0,('N_clusters','Silhouette_score','Davies_bouldin_score'))
total_rows = len(lst)
total_columns = len(lst[0])
label_title_1 = Label(window, text='K_Means', font=Font_title, fg= Text_color, bg=Bg_Color, padx=20, pady=10).grid(row= 0,column=2)

def create_table():
    for i in range(total_rows):
        for j in range(total_columns):
            a = Entry(window, width=20,font=('Times New Roman',16))
            a.grid(row = i+1, column= j+1)#,padx= (20,0) if j == 0 else 0
            a.insert(END, lst[i][j])

label_msg_1 = Label(window, text=f'Số cụm để mô hình tốt nhất là: {n_clusters} \n Silhouette: {max_score} \n Davies_bouldin: {davies}', font=Font_text, fg= '#fa0606', bg='#E2F7B5', pady=10).grid(row = 12, column=0, columnspan=3, sticky='wsen')

label_title_2 = Label(window, text='Dự đoán', font=Font_title, fg= Text_color, bg= Bg_Color, padx=20, pady=10).grid(row= 13,column=2)

lable_age = Label(window, text = "Age", font=Font_text, fg= Text_color, bg='#E2F8B5').grid(row = 14, column = 0, padx=10, pady = 10,sticky='e')
text_age = Entry(window, width=13)
text_age.insert(0, data_Test[0][0])
text_age.grid(row=14, column=1, sticky='w')

lable_gender = Label(window, text = "Gender", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 14, column = 1, padx=10, pady = 10, sticky='e')
values_gender = [0,1]
gender_choosen = ttk.Combobox(window, width=10, values=values_gender)
gender_choosen.insert(0,data_Test[0][1])
gender_choosen.grid(row=14, column=2, sticky='w')

lable_chest_pain_type = Label(window, text = "Chest pain type", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 14, column = 3, padx=10, pady = 10,sticky='e')
values_chest_pain_type = [0,1,2,3,4]
chest_pain_type_choosen = ttk.Combobox(window, width=10, values=values_chest_pain_type)
chest_pain_type_choosen.insert(0, data_Test[0][2])
chest_pain_type_choosen.grid(row=14, column=4, sticky='w')

lable_blood_pressure = Label(window, text = "Blood pressure", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 15, column = 0, padx=10, pady = 10, sticky='e')
text_blood_pressure = Entry(window, width=13)
text_blood_pressure.insert(0, data_Test[0][3])
text_blood_pressure.grid(row=15, column=1, sticky='w')

lable_cholesterol = Label(window, text = "Cholesterol", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 15, column = 1, padx=10, pady = 10, sticky='e')
text_cholesterol = Entry(window, width=13)
text_cholesterol.insert(0, data_Test[0][4])
text_cholesterol.grid(row=15, column=2, sticky='w')

lable_max_heart_rate = Label(window, text = "Max heart rate", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 15, column = 3, padx=10, pady = 10, sticky='e')
text_max_heart_rate = Entry(window, width=13)
text_max_heart_rate.insert(0, data_Test[0][5])
text_max_heart_rate.grid(row=15, column=4, sticky='w')

lable_exercise_angina = Label(window, text = "Exercise angina", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 16, column = 0, padx=10, pady = 10,sticky='e')
values_exercise_angina = [0,1]
exercise_angina_choosen = ttk.Combobox(window, width=10, values=values_exercise_angina)
exercise_angina_choosen.insert(0, data_Test[0][6])
exercise_angina_choosen.grid(row=16, column=1, sticky='w')

lable_plasma_glucose = Label(window, text = "Plasma glucose", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 16, column = 1, padx=10, pady = 10,sticky='e')
text_plasma_glucose = Entry(window, width=13)
text_plasma_glucose .insert(0, data_Test[0][7])
text_plasma_glucose .grid(row=16, column=2, sticky='w')

lable_skin_thickness = Label(window, text = "Skin thickness", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 16, column = 3, padx=10, pady = 10, sticky='e')
text_skin_thickness= Entry(window, width=13)
text_skin_thickness.insert(0, data_Test[0][8])
text_skin_thickness.grid(row=16, column=4, sticky='w')

lable_insulin = Label(window, text = "Insulin", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 17, column = 0, padx = 10, sticky='e')
text_insulin = Entry(window, width=13)
text_insulin.insert(0, data_Test[0][9])
text_insulin.grid(row = 17, column = 1, sticky='w')

lable_bmi = Label(window, text = "Bmi", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 17, column = 1, padx=10, pady = 10, sticky='e')
text_bmi = Entry(window, width=13)
text_bmi.insert(0, data_Test[0][10])
text_bmi.grid(row=17, column=2, sticky='w')

label = Label(window, text = "Diabetes pedigree", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row=17, column=3, pady=10, sticky='e')
text_diabetes_pedigree = Entry(window, width=13)
text_diabetes_pedigree.insert(0, data_Test[0][11])
text_diabetes_pedigree.grid(row=17, column=4, sticky='w')

lable_hypertension = Label(window, text = "Hypertension", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 18, column = 0, padx=10, pady = 10,sticky='e')
values_hypertension = [0,1]
hypertension_choosen = ttk.Combobox(window, width=10, values=values_hypertension)
hypertension_choosen.insert(0, data_Test[0][12])
hypertension_choosen.grid(row=18, column=1, sticky='w')

lable_heart_disease = Label(window, text = "Heart disease", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 18, column = 1, padx=10, pady = 10,sticky='e')
values_heart_disease = [0,1]
heart_disease_choosen = ttk.Combobox(window, width=10, values=values_heart_disease)
heart_disease_choosen.insert(0, data_Test[0][13])
heart_disease_choosen.grid(row=18, column=2, sticky='w')

lable_residence_type = Label(window, text = "Residence type", font=Font_text, fg= Text_color, bg= Bg_Color).grid(row = 18, column = 3, padx=10, pady = 10,sticky='e')
values_residence_type = [0,1]
residence_type_choosen = ttk.Combobox(window, width=10, values=values_residence_type)
residence_type_choosen.insert(0, data_Test[0][14])
residence_type_choosen.grid(row=18, column=4, sticky='w')

def predict_label():
    age = text_age.get()
    gender = gender_choosen.get()
    chest_pain_type = chest_pain_type_choosen.get()
    blood_pressure = text_blood_pressure.get()
    cholesterol = text_cholesterol.get()
    max_heart_rate = text_max_heart_rate.get()
    exercise_angina = exercise_angina_choosen.get()
    plasma_glucose = text_plasma_glucose .get()
    skin_thickness = text_skin_thickness.get()
    insulin = text_insulin.get()
    bmi = text_bmi.get()
    diabetes_pedigree = text_diabetes_pedigree.get()
    hypertension = hypertension_choosen.get()
    heart_disease = heart_disease_choosen.get()
    residence_type = residence_type_choosen.get()
    # residence_type = (0 if (residence_type == '0 - Urbal') else 1)
    if((age == '') | (gender == '')  | (chest_pain_type == '') | (blood_pressure == '') | (cholesterol == '') | (max_heart_rate == '') | (exercise_angina == '') | (plasma_glucose == '') | (skin_thickness == '') | (insulin == '') | (bmi == '')
       | (diabetes_pedigree == '') | (hypertension == '')| (heart_disease == '')| (residence_type == '') ):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        # kmeans = KMeans(n_clusters= n_clusters, n_init='auto').fit(data_Train)
        dt_test = np.array([float(age),float(gender),float(chest_pain_type),float(blood_pressure),float(cholesterol),float(max_heart_rate),float(exercise_angina),float(plasma_glucose),float(skin_thickness),float(insulin),float(bmi), float(diabetes_pedigree), float(hypertension), float(heart_disease),float(residence_type)]).reshape(1,-1)
        # print(dt_test)
        label = predict_cluster(dt_test, centroids_best)
        messagebox.showinfo("Kết quả dự đoán", f'{label}')

btn_predict = Button(window, text= 'Kết quả', font= Font_text, bg='#FDFF0C',fg = Text_color, borderwidth=1, relief='solid',activebackground='#08FF35', command=predict_label, padx=10, pady=5).grid(row=22, column=2, pady=10)

setup_window()
create_table()

window.mainloop()
