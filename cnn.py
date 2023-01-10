import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img
from keras.models import load_model

import cv2

train_dir = 'C:/Users/mehmet/desktop/train' # Bilgisayardaki train verilerin yolu
test_dir = 'C:/Users/mehmet/desktop/test' # Bilgisayardaki test verilerin yolu

class_name = glob(train_dir + '/*') # train verilerinde ki klasorlerin ismi
print(class_name)

number_of_class = len(class_name) # train verilerinde ki klasorlerin sayisi
print(number_of_class)


class Model(object):
    
    def __init__(self):
        """
        Initializer Metodumuz.
        Kullanacagimiz degerlerin tanimlanmasi ve ilk deger atamasini
        burda yaptik.
        """
        
        self.batch_size = 32
        self.model = None
        self.train_datagen = None
        self.test_datagen = None
        self.train_generator = None
        self.test_generator = None
        self.IMAGE_WIDTH = 180
        self.IMAGE_HEIGHT = 180
        self.epochs = 50
    
    def imageGenerator(self, train_dir, test_dir):
        """
        Resim Uretme Metodumuz.

        Parameters
        ----------
        train_dir : string
            Bilgisayardaki egitim verilerinin yolu.
        test_dir : string
            Bilgisayardaki test/validation verilerinin yolu.
        """
        
        self.train_datagen = ImageDataGenerator(rescale=1./255,
                                   zoom_range=0.2,
                                   rotation_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

        self.test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = self.train_datagen.flow_from_directory(directory=train_dir,
                                                            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
                                                            class_mode='categorical',
                                                            color_mode='rgb',
                                                            batch_size=self.batch_size)
        
        self.test_generator = self.test_datagen.flow_from_directory(directory=test_dir,
                                                          target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
                                                          class_mode='categorical',
                                                          color_mode='rgb',
                                                          batch_size=self.batch_size)
    
    def modelCreate(self, number_of_class):
        """
        Modelimizi Olusturdugumuz Metodumuz.

        Parameters
        ----------
        number_of_class : int
            Siniflandirma yapacagimiz siniflarin sayisi.
        """

        self.model = Sequential()
        
        self.model.add(Conv2D(filters=16, kernel_size=(3, 3),
                              input_shape=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Dropout(0.25))
        
        # self.model.add(Conv2D(filters=64, kernel_size=(3, 3)))
        # self.model.add(BatchNormalization())
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(2, 2))
        
        # self.model.add(Conv2D(filters=128, kernel_size=(3, 3)))
        # self.model.add(BatchNormalization())
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(2, 2))
        self.model.add(Dropout(0.25))
        
        # self.model.add(Conv2D(filters=256, kernel_size=(3, 3)))
        # self.model.add(Activation('relu'))
        # self.model.add(MaxPooling2D(2, 2))
        
        self.model.add(Flatten())
        
        # self.model.add(Dense(units=256))
        # self.model.add(Activation('relu'))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.5))
        
        self.model.add(Dense(units=512))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        
        self.model.add(Dense(units=number_of_class))
        self.model.add(Activation('softmax'))
        
        self.model.summary()
    
    
    def modelCompile(self):
        """
        Modelimizi Derleme Metodumuz.
        """
        
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def modelFit(self):
        """
        Modelimizi Train Ettigimiz Metodumuz.
        """
        
        self.model.fit_generator(generator=self.train_generator,
                                 epochs=self.epochs,
                                 validation_data=self.test_generator)
    
    def visualization(self):
        """
        Modelin Tahminlerinin Görselleştirilmesi.
        
        Accuracy ile Validation Accuracy'i ve
        Loss ile Validation Loss degerlerini gorsellestirdigimiz metodumuz.
        """
    
        plt.plot(self.model.history.history['accuracy'], label='Accuracy')
        plt.plot(self.model.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.show()
        
        plt.plot(self.model.history.history['loss'], label='Loss')
        plt.plot(self.model.history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.show()


    def resultPredict(self):
        """
        Modelin Tahmin Metodu.
        
        Model'e input olarak bilgisayardaki yolunu girdigimiz ve
        girdigimiz resmin on islemden gecirdigimiz metodumuz.
        """
        
        image = load_img(input('Lutfen Resmin Yolunu Giriniz: '),
                         target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        plt.imshow(image)
        plt.show()
        print('Resmin Boyutlari: ', image.size)
        
        image_array = img_to_array(image)
            
        image_array = image_array.reshape((1, ) + image_array.shape)
            
        result = self.model.predict(image_array)
        #print('Tahmin Sonucu:\n', result)
        
        result_arr = []
        for i in range(3):
            result_arr.append(result[0][i])
            
        #print(result_arr)
            
        categories = ['catlak', 'damarli', 'iyi']
            
        for i in range(3):
            print(f'{categories[i]}: ' , f'{result_arr[i] * 100:.7f}%')
    
    
    def saveModel(self):
        """
        Modeli Kayıt Ettiğimiz Metod.
        """
        
        self.model.save('my_model')
        self.model.save_weights('my_model_weights.h5')
    
    
    def loadModel(self):
        """
        Kaydedilmiş Modeli Eğitmeden Kullanmak İçin Gereken Metod.
        """
        model1 = load_model('my_model')
        
        image = load_img(input('Lutfen Resmin Yolunu Giriniz: '),
                         target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        plt.imshow(image)
        plt.show()
        print('Resmin Boyutlari: ', image.size)
        
        image_array = img_to_array(image)
            
        image_array = image_array.reshape((1, ) + image_array.shape)
            
        result = model1.predict(image_array)
        #print('Tahmin Sonucu:\n', result)
        
        result_arr = [] 
        for i in range(3):
            result_arr.append(result[0][i])
            
        #print(result_arr)
            
        categories = ['catlak', 'damarli', 'iyi']
            
        for i in range(3):
            print(f'{categories[i]}: ' , f'{result_arr[i] * 100:.7f}%')


class ColorDetection(object):
    
    def __init__(self):
        self.cam = None
        self.model2 = None
        self.prediction = None
        self.categories = ['catlak', 'damarli', 'iyi']


    def readFromCamera(self):
        
        self.cam = cv2.VideoCapture(0) # kamerayi ac 0 pc kamerasi, 1 harici kamera
        
        self.model2 = load_model('my_model') # kaydettigimiz modeli yukluyoruz
        
        while True: # Kameranin surekli acik olmasi icin sonsuz dongu
            # ret=kamera acildi(Bool)
            # image=ekrandaki goruntunun matrixi
            ret, image = self.cam.read()
            image = cv2.resize(image, (224, 224)) # goruntuyu yeniden boyutlandiriyoruz
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # renk olarak BGR2RGB formatina ayarliyoruz
            
            tensor_image = np.expand_dims(image, axis=0) # fazladan bir katman ekliyoruz
            
            self.prediction = self.model2.predict(tensor_image) # tahmin ettiriyoruz
            
            cv2.imshow('Frame: ', image) # ekranda gosteriyoruz
            
            # 50'ms de tusa basmayi kontrol et
            # x tusuna basilinca programi sonlandir
            if cv2.waitKey(50) & 0xFF == ord('x'):
                break
        
        self.cam.release() # kamerayi kapat
        cv2.destroyAllWindows() # tum acik pencereleri kapat


model = Model()

model.imageGenerator(train_dir, test_dir)
model.modelCreate(number_of_class)
model.modelCompile()
model.modelFit()
model.saveModel()
model.visualization()
model.resultPredict()

# model.loadModel()


cd = ColorDetection()

# cd.readFromCamera()

