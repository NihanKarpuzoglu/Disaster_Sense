import os 
import cv2
import numpy as np
from PIL import Image

def divide_data(img_data, binary_out_arr, train_percentage):
    num_data=np.array(img_data).shape[0]
    
    train_size=num_data*train_percentage//100
   
    train_data=img_data[:train_size]
    train_binary_out=binary_out_arr[:train_size]
    
    test_data=img_data[train_size:]
    test_binary_out=binary_out_arr[train_size:]
    
    train_binary_out=np.array(train_binary_out, ndmin=2)
    test_binary_out=np.array(test_binary_out,ndmin=2)
    
    return train_data, test_data, train_binary_out, test_binary_out

def get_images(disaster_df):
    wildfire_img=[]
    earthquake_img=[]
    hurricane_img=[]
    flood_img=[]
    
    wildfire_out=[]
    earthquake_out=[]
    hurricane_out=[]
    flood_out=[]
    for index, disaster in disaster_df.iterrows():
        img_path = disaster['image_path']
        img="C:/Users/Nihan/Downloads/CrisisMMD_v2.0/"+str(img_path)
        pic = cv2.imread(img)
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
        img_array = Image.fromarray(pic, 'RGB')
        img_rs = img_array.resize((80,80))
        img_rs = np.array(img_rs)
        
        if "hurricane_irma" in img_path:
            hurricane_img.append(img_rs)
            hurricane_out.append(disaster['is_disaster'])
        elif "hurricane_maria" in img_path:
            hurricane_img.append(img_rs)
            hurricane_out.append(disaster['is_disaster'])
        elif "hurricane_harvey" in img_path:
            hurricane_img.append(img_rs)
            hurricane_out.append(disaster['is_disaster'])
        elif "california_wildfires" in img_path:
            wildfire_img.append(img_rs)
            wildfire_out.append(disaster['is_disaster'])
        elif "iraq_iran_earthquake" in img_path:
            earthquake_img.append(img_rs)
            earthquake_out.append(disaster['is_disaster'])
        elif "mexico_earthquake" in img_path:
            earthquake_img.append(img_rs)
            earthquake_out.append(disaster['is_disaster'])
        elif "srilanka_floods" in img_path:
            flood_img.append(img_rs)
            flood_out.append(disaster['is_disaster'])
    return (hurricane_img, wildfire_img, earthquake_img, flood_img, hurricane_out, wildfire_out, earthquake_out, flood_out)