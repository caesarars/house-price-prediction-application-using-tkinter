import tkinter as tk
""" GUI Application Library """
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *

""" Machine Learning Library """
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

""" Global Variable """
globalFont = 'Sans'
X_full = None
X_manual_full = None
X_test_full = None
X_test_manual_full = None
y = None
cat_cols = None
num_cols = None
cols_with_missing = None
X_cormat = None
path_file_x = None
path_file_x_test = None
X_train_full = None
X_valid_full = None
X_train = None
X_valid = None
y_train = None
y_valid = None
X_test = None
bestIndexPipe = None
bestIndexBoost = None

Xpipe = []
Ypipe = []

Xboost = []
Yboost = []

class App(tk.Tk):
   global bestIndexPipe
   def __init__(self, *args, **kwargs):
    tk.Tk.__init__(self, *args, **kwargs)
    container = tk.Frame(self)
    container.pack(side="top", fill="both", expand=True)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    self.b = StringVar(self)
    self.bestIndexP = StringVar(self)
    self.bestIndexP.set("0")
    self.bestMAEPipe = StringVar(self)
    self.dataTrain = StringVar(self)
    self.dataTest = StringVar(self)
    self.dataSizeTrain = StringVar(self)
    self.dataSizeTest = StringVar(self)

    self.frames = {}
    for F in (PageOne,PageTwo,PageThree,PageFour,PageFive):
        page_name = F.__name__
        frame = F(parent=container, controller=self)
        self.frames[page_name] = frame

        # put all of the pages in the same location;
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")

    self.geometry("800x600")
    self.title("Machine Learning")

    self.show_frame("PageOne")
   def show_frame(self, page_name):
       '''Show a frame for the given page name'''
       frame = self.frames[page_name]
       frame.tkraise()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        self.controlller = controller
        tk.Label(self, text="M A I N  M E N U", font=globalFont,padx=80).pack(pady=20)
        tk.Button(self, text="Dataset", font=globalFont,width=12,bg="#f6f6f6",
                  command=lambda: controller.show_frame("PageTwo")).pack( pady=12)
        tk.Button(self, text="Features", font=globalFont,width=12,bg="#f6f6f6",
                  command=lambda:controller.show_frame("PageThree")).pack( pady=12)
        tk.Button(self, text="Performance", font=globalFont,width=12,bg="#f6f6f6",
                  command=lambda: controller.show_frame("PageFour")).pack( pady=12)
        tk.Button(self, text="Output", font=globalFont,width=12,bg="#f6f6f6",
                  command=lambda: controller.show_frame("PageFive")).pack(pady=12)


class PageTwo(tk.Frame):
    def __init__(self, parent,controller):
        tk.Frame.__init__(self, parent)
        self.controlller = controller
        tk.Label(self,text="Insert dataset in this menu",font=globalFont).pack(padx  = 20,pady = 20)
        tk.Button(self,text="Training data",width=12,
                  command=lambda:self.__insertTrainData(controller),bg='#f6f6f6').pack(padx= 10, pady = 10)
        tk.Button(self,text="Testing data",width=12,
                  command=lambda:self.__insertTestData(controller),bg='#f6f6f6').pack(padx=10,pady=10)
        tk.Label(self, text="Training Dataset", font=globalFont).pack(pady=20, padx=10)
        tk.Label(self, textvariable=controller.dataTrain).pack(pady=5, padx=10)
        tk.Label(self,textvariable=controller.dataSizeTrain).pack(padx=10,pady=5)
        tk.Label(self, text="Testing Dataset", font=globalFont).pack(pady=20, padx=10)
        tk.Label(self, textvariable=controller.dataTest).pack(pady=5, padx=10)
        tk.Label(self, textvariable=controller.dataSizeTest).pack(padx=10, pady=5)
        tk.Button(self, text="Main menu", width=10,
                  command=lambda: controller.show_frame("PageOne"),bg='#f6f6f6').pack(padx=10, pady=10,side=BOTTOM)

    def __insertTrainData(self,controller):
        global X_full,path_file_x,X_cormat,X_manual_full
        try:
            x = tk.filedialog.askopenfilename(initialdir = "/",
                                              title = "Select file",
                                              filetypes = (("csv files","*.csv"),("all files","*.*")))
            path_file_x = x;
            controller.dataTrain.set(x)
            X_full = pd.read_csv(x)
            X_manual_full = pd.read_csv(x)
            X_cormat = X_full.corr()
            controller.dataSizeTrain.set(str(X_full.shape))
            print(sum(X.column))
        except Exception as e:
            print("something went error ---> ", e)
    def __insertTestData(self,controller):
        global X_test_full,X_test_manual_full
        try :
            test = tk.filedialog.askopenfilename(initialdir = "/",
                                                 title = "Select file",
                                                 filetypes = (("csv files","*.csv"),("all files","*.*")))
            controller.dataTest.set(test)
            X_test_full = pd.read_csv(test)
            X_test_manual_full = pd.read_csv(test)
            controller.dataSizeTest.set(str(X_test_full.shape))
            print(X_test_full.head())
        except Exception as e :
            print("something went error ---> ", e)


class PageThree(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self, parent)
        self.controlller = controller
        tk.Label(self,text="Processing Data",font='Sans').pack(padx = 20, pady = 20)
        tk.Button(self,text="Features",command=lambda:self.features()).pack(padx = 10, pady =10)
        tk.Button(self, text="Back to menu", width=10,
                  command=lambda: controller.show_frame("PageOne")).pack(padx=10, pady=10,side=BOTTOM)

    def features(self):
        global y,X_full,X_test_full,cols_with_missing,X_train_full,X_train
        global X_valid_full,X_valid,y_train,y_valid,cat_cols,num_cols
        global selectedCat_cols,selectedNum_cols,selectedCat_cols,X_test,X_cormat

        colsCorrmat = X_cormat.nlargest(15, 'SalePrice')['SalePrice'].index
        X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
        y = X_full.SalePrice
        X_full.drop(['SalePrice'], axis=1, inplace=True)
        cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()]
        print(cols_with_missing)
        X_full.drop(cols_with_missing, axis=1, inplace=True)
        X_test_full.drop(cols_with_missing, axis=1, inplace=True)
        FrameColumn = tk.LabelFrame(self, text="The Columns",padx=10, pady=10,)
        FrameColumn.pack()
        """ COLUMN WITH MISSING"""
        frameMiss = tk.LabelFrame(FrameColumn,text="Column with missing value",padx=5,pady=5)
        frameMiss.pack(anchor=N,side=LEFT,padx=10,pady=10)
        """ LIST BOX FOR COLUMN WITH MISSING VALUE """
        listMiss = Listbox(frameMiss,selectmode = MULTIPLE)
        for i in range(len(cols_with_missing)):
            listMiss.insert(i,str(cols_with_missing[i]))
        """ SPLITNG DATASET INTO TRAINING DATASET AND TESTING DATA SET """
        X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                              train_size=0.8, test_size=0.2,random_state=0)
        """ SELECT THE CATEGORICAL COLUMN """
        categorical_cols = [cname for cname in X_train_full.columns if
                            X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"]
        """ SELLECT THE NUMERICAL COLUMN """
        numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
        cat_cols = categorical_cols
        num_cols = numerical_cols
        """JOIN NUMERICAL AND CATEGORICAL COLUMN"""
        my_cols = categorical_cols+numerical_cols

        X_train = X_train_full[my_cols].copy()
        X_valid = X_valid_full[my_cols].copy()
        X_test = X_test_full[my_cols].copy()

        frameCategorical = tk.LabelFrame(FrameColumn,text="categorical column",padx = 5 , pady= 5)
        frameNumerical = tk.LabelFrame(FrameColumn,text="numerical column", padx=5, pady=5)
        frameCategorical.pack(anchor=N,side=LEFT,padx=10,pady=10)
        frameNumerical.pack(anchor=N,side=LEFT,padx=10,pady=10)
        scrollbar1 = Scrollbar(frameNumerical)
        scrollbar2 = Scrollbar(frameCategorical)
        scrollbar3 = Scrollbar(frameMiss)
        scrollbar1.pack(side=RIGHT,fill=Y)
        scrollbar2.pack(side=RIGHT,fill=Y)
        scrollbar3.pack(side=RIGHT,fill=Y)

        listCat = Listbox(frameCategorical,selectmode = MULTIPLE)
        listNum = Listbox(frameNumerical,selectmode = MULTIPLE)

        listNum.config(yscrollcommand=scrollbar1.set)
        scrollbar1.config(command=listNum.yview)
        listCat.config(yscrollcommand=scrollbar2.set)
        scrollbar2.config(command=listCat.yview)
        listMiss.config(yscrollcommand=scrollbar3.set)
        scrollbar3.config(command=listMiss.yview)

        listCat.pack()
        listNum.pack()
        listMiss.pack()
        for i in range(len(categorical_cols)):
            listCat.insert(i,str(categorical_cols[i]))
        for i in range(len(numerical_cols)):
            listNum.insert(i,str(numerical_cols[i]))
        btnNum = Button(frameNumerical, text="save", command=lambda: saveNumCol(listNum.curselection()))
        btnNum.pack()

        btnCat = Button(frameCategorical,text="save",command = lambda:saveCatCol(listCat.curselection()))
        btnCat.pack()

        def saveNumCol(item):
            global num_cols
            cols = []
            for i in range(len(item)):
                cols.append(listNum.get(i))
            print(cols)
            num_cols = cols
            print(num_cols)

        def saveCatCol(item):
            global cat_cols
            cols = []
            for i in range(len(item)):
                cols.append(listCat.get(i))
            print(cols)
            cat_cols = cols
            print(cat_cols)





class PageFour(tk.Frame):
    PrepPipe=None
    def __init__(self,parent,controller):
        tk.Frame.__init__(self, parent)
        self.controlller = controller

        tk.Label(self,text="Mean Absolute Error",font=globalFont).pack(padx=25,pady=25,anchor=tk.W)
        tk.Button(self,text="Mean Absolute Error Pipeline",bg='#f6f6f6',width=25,
                  command=lambda:self.MAEforPipeline(entr.get(),incr.get(),controller)).pack(padx=10,pady=10)
        tk.Button(self, text="Back to Menu", width=10,
                  command=lambda: controller.show_frame("PageOne")).pack(padx=10, pady=10,side=BOTTOM)
        entr = tk.Entry(self,bg='#f6f6f6')
        incr = tk.Entry(self,bg='#f6f6f6')
        tk.Label(self,text='Increment').pack()
        incr.pack(padx=10,pady=10)
        tk.Label(self,text='Max Trees').pack()
        entr.pack(padx=10,pady=10)
        tk.Label(self,text="Best MAE for Pipeline",font=globalFont).pack(padx=10,pady=10)
        tk.Label(self,textvariable=controller.bestMAEPipe).pack(padx=10, pady=10)


    def MAEforPipeline(self,n,incr,controller):
        global Xpipe,Ypipe,bestIndexPipe
        i = 0
        x = []
        y = []
        i += int(incr)
        j = int(n)
        inc = int(incr)
        numerical_transformer = SimpleImputer(strategy="constant")
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )
        while i <= j:
               model = RandomForestRegressor(n_estimators=i,random_state=1,max_depth=20)
               clf = Pipeline(steps=[
                   ('preprocessor', preprocessor),
                   ('model', model)
               ])
               x.append(i)
               clf.fit(X_train, y_train)
               predsPipeline = clf.predict(X_valid)
               mae = mean_absolute_error(y_valid,predsPipeline)
               y.append(mae)
               Xpipe = x
               Ypipe = y
               print('MAE : ', mae)
               i+=inc
        print(Xpipe)
        print(Ypipe)
        bestMAE = min(y)
        controller.bestMAEPipe.set(str(bestMAE))
        for i in range(len(y)):
            if y[i] == bestMAE:
                bestIndexPipe = i
                controller.bestIndexP.set("Best number of Trees for Pipeline "+str(x[i]))
        print(controller.bestIndexP.get())
        plt.title("Machine Learning")
        plt.plot(x,y,label="Pipeline process")
        plt.xlabel("n_estimators")
        plt.ylabel("MAE")
        plt.title("Trees vs MAE")
        plt.legend()
        plt.show()


class PageFive(tk.Frame):
    global Xpipe,Xboost,Ypipe,Yboost,X_test_full
    def __init__(self,parent,controller):
        tk.Frame.__init__(self, parent)
        self.controlller = controller
        framePipe = tk.LabelFrame(self,text="Pipeline",padx=150, pady=5, )
        framePipe.pack()

        tk.Label(framePipe, textvariable=controller.bestIndexP).pack(padx=10, pady=10)
        n_estimate = tk.Entry(framePipe)
        tk.Label(framePipe,text='Trees').pack()
        n_estimate.pack(padx=10, pady=10)
        tk.Button(framePipe, text="Process",
                  command=lambda:self.predPipe(n_estimate.get())).pack(padx=10, pady=10)
        tk.Button(self, text="Main menu", width=10,
                  command=lambda: controller.show_frame("PageOne")).pack(padx=10, pady=10, side=BOTTOM)


        manualInpuFrame = tk.LabelFrame(self, text='Manual Input', )
        manualInpuFrame.pack()

        inputFrame = tk.Frame(manualInpuFrame, padx=60)
        inputFrame.pack(side=LEFT)

        outputFrame = tk.Frame(manualInpuFrame, padx=60)
        outputFrame.pack(side=LEFT)

        tk.Label(inputFrame, text='Id').pack()
        id = tk.Entry(inputFrame)
        id.pack()
        tk.Label(inputFrame, text='overallQual (1 - 10)').pack()
        overallQual = tk.Entry(inputFrame)
        overallQual.pack()
        tk.Label(inputFrame, text='GrLivArea (334-5642) ft²').pack()
        GrLivArea = tk.Entry(inputFrame)
        GrLivArea.pack()
        tk.Label(inputFrame, text='GarageCars (0 - 4)').pack()
        GarageCars = tk.Entry(inputFrame)
        GarageCars.pack()
        tk.Label(inputFrame, text='GarageArea (0 - 1418) ft²').pack()
        GarageArea = tk.Entry(inputFrame)
        GarageArea.pack()
        tk.Button(inputFrame, text='Proses',
                  command=lambda: self.manual_proses(id.get(),
                                                     overallQual.get(),
                                                     GrLivArea.get(),
                                                     GarageCars.get(),
                                                     GarageArea.get())).pack(pady=5)


    def predPipe(self,n_estimator):
        global bestPipe, Xpipe, Ypipe, bestIndexPipe
        numerical_transformer = SimpleImputer(strategy="constant")
        # Preprocessing for cateogrical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num_cols),
                ('cat', categorical_transformer, cat_cols)
            ]
        )
        model = RandomForestRegressor(n_estimators=int(n_estimator), random_state=1,max_depth=20)
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        clf.fit(X_train, y_train)
        predsPipeline = clf.predict(X_test)
        output = pd.DataFrame({
            'Id':X_test.Id,
            'SalePrice':predsPipeline

        })
        output.to_csv('app_output_test.csv',index=False)
        export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
        output.to_csv(export_file_path, index=False, header=True)

    def manual_proses(self,id,overallQual,GrLivArea,GarageCars,GarageArea):
        global bestRelationCols
        param = [id,overallQual,GrLivArea,GarageCars,GarageArea]
        print(param)
        column = []
        maxV = []
        for i in X_test_manual_full.columns:
            column.append(i)
            item_counts = X_test_manual_full[i].value_counts(sort=False)
            top = item_counts.loc[[item_counts.idxmax()]]
            value = top.index[0]
            maxV.append(value)
        z = dict(zip(column, maxV))
        """ THE MOST FREQUENT VALUE IN COLUMN"""
        print(len(column))
        predictValue = []
        for i in z:
            if i =='Id':
                predictValue.append(param[0])
            elif i == 'OverallQual':
                predictValue.append(param[1])
            elif i == 'GrLivArea':
                predictValue.append(param[2])
            elif i == 'GarageCars':
                predictValue.append(param[3])
            elif i == 'GarageArea':
                predictValue.append(param[4])
            else :
                predictValue.append(z.get(i))
        print(len(predictValue))

        from csv import writer
        def append_list_as_row(file_name, list_of_elem):
            # Open file in append mode
            with open(file_name, 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(list_of_elem)
        #memanggil fungsi append_list_as_row
        append_list_as_row("manualInput.csv", predictValue)

        """ PIPELINE METHOD WITH NEW CSV"""
        X_full_manual = pd.read_csv('train.csv')
        X_test_full_manual = pd.read_csv('manualInput.csv')
        X_full_manual.dropna(axis=0, subset=['SalePrice'], inplace=True)
        y_manual = X_full_manual.SalePrice
        X_full_manual.drop(['SalePrice'], axis=1, inplace=True)
        X_train_full_m, X_valid_full_m, y_train_m, y_valid_m = train_test_split(X_full_manual, y_manual,
                                                                        train_size=0.8, test_size=0.2,
                                                                        random_state=0)
        categorical_cols_m = [cname for cname in X_train_full_m.columns if
                            X_train_full_m[cname].nunique() < 10 and
                            X_train_full_m[cname].dtype == "object"]
        numerical_cols_m = [cname for cname in X_train_full_m.columns if
                          X_train_full_m[cname].dtype in ['int64', 'float64']]

        my_cols = categorical_cols_m + numerical_cols_m
        X_train_m = X_train_full_m[my_cols].copy()
        X_test_m = X_test_full_manual[my_cols].copy()
        numerical_transformer = SimpleImputer(strategy="constant")
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols_m),
                ('cat', categorical_transformer, categorical_cols_m)
            ]
        )
        model = RandomForestRegressor(n_estimators=int(bestIndexPipe),
                                      random_state=1,max_depth=20  )
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        clf.fit(X_train_m, y_train_m)
        predsPipeline = clf.predict(X_test_m)
        print(predsPipeline)
        output = pd.DataFrame({
            'Id': X_test_m.Id,
            'SalePrice': predsPipeline
        })
        id = 'Id : ' + str(output['Id'].values[-1])
        salePrice = 'SalePrice : ' + str(output['SalePrice'].values[-1])
        manual_output = id+ ' --> '+ salePrice
        print(manual_output)
        tk.messagebox.showinfo(title='Hasil prediksi manual input',message=manual_output)

if __name__ =="__main__":
    app = App()
    app.geometry("1080x720")
    app['bg'] = '#49A'
    app.mainloop()
