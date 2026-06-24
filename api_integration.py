from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field
from typing import Literal,Annotated
import pandas as pd
import pickle
from processing import weekday_and_weekend, demojizeing_notification , NLP
from db_config import mongoDB

import asyncio

#collection

notif_ai_collection = mongoDB['notification_data']

retraining_in_progress = False

with open('feature_order_model2_for_retrain.pkl','rb') as f:
    feature_order_model2_for_retrain = pickle.load(f)

#with open('test_retrain_model.pkl','rb') as f:
#    test_retrain_model = pickle.load(f)

with open('model1.pkl','rb') as f:
    model1 = pickle.load(f)

with open('model2.pkl','rb') as f:    
     model2 = pickle.load(f)

with open('feature_order_model1.pkl','rb') as f:
    feature_order_model1 = pickle.load(f)

with open('feature_order_model2.pkl','rb') as f:
    feature_order_model2 = pickle.load(f)

with open('oer_app.pkl','rb') as f:
    oer_app = pickle.load(f)

with open('onehot_encoder_app.pkl','rb') as f:
    oe_app = pickle.load(f)

with open('onehot_encoder_category.pkl','rb') as f:
    oe_category = pickle.load(f)

with open('onehot_encoder_model1_priority_label.pkl','rb') as f:
    oe_model1_priority_label = pickle.load(f)

with open('onehot_encoder_time.pkl','rb') as f:
    oe_time = pickle.load(f)

with open('tfidf_vectorizer.pkl','rb') as f:
    vector = pickle.load(f) 



app = FastAPI()

model2_lock = asyncio.Lock()
app.state.latest_prediction = {
    "app_name": "None",
    "notification_text": "Waiting for data...",
    "final_prediction": 0.0
}


class notification(BaseModel):
    notification_text: Annotated[str,Field(description="the text if the notifcation")]
    app_name: Annotated[str,Field(description="the app name of the notification")]
    category: Annotated[
    Literal["social", "message", "productivity", "email", "shopping", "work"],
    Field(description="Functional category of the notification")]
    message_length:Annotated[int,Field(description="the length of the notification text")]
    contains_urgent_words: Annotated[bool,Field(description="Indicates whether the notification text contains urgent keywords such as 'urgent', 'important', 'asap', 'immediately', or 'action required'. (1 = Yes, 0 = No)")]
    time_of_day:Annotated[str,Field(description="Time period when the notification was received (Morning, Afternoon, Evening, or Night).")]
    weekday:Annotated[str,Field(description="the day of the week")]
    screen_on: Annotated[bool,Field(description="if the screen is on during the notification arrive or not ,yes = 1 and no=0")]
    battery_level:Annotated[int,Field(description="the battery percentage of the phone when the notification arrived")]
    headset_connected:Annotated[bool,Field(description="Whether the head set was connected when the notification arrived yes =1 and no = 0")]
    notification_frequency_today:Annotated[int , Field(description="Number of notifications received from the same app today.")]
    app_usage_minutes_today:Annotated[int, Field(description="Total minutes the user spent using the app today")]
    app_open_frequency:Annotated[int , Field(description="Number of times the user opened the app today")]
    notification_clicked:Annotated[bool,Field(description="Whether the previous notification from the same app was clicked.")]
    notification_dismissed:Annotated[bool,Field(description="Whether the previous notification from the same app was dismissed.")]

latest_prediction_store = {
    "app_name": "None",
    "notification_text": "Waiting for data...",
    "final_prediction": 0.0}

@app.post('/predict')
async def predict_priority(data:notification):
    try:
        input_df = pd.DataFrame([{
        'notification_text':data.notification_text,
        'app_name':data.app_name,
        'category':data.category,
        'message_length':data.message_length,
        'contains_urgent_words':data.contains_urgent_words,
        'time_of_day':data.time_of_day,
        'weekday':data.weekday,
        'screen_on':data.screen_on,
        'battery_level':data.battery_level,
        'headset_connected':data.headset_connected,
        'notification_frequency_today':data.notification_frequency_today,
        'app_usage_minutes_today':data.app_usage_minutes_today,
        'app_open_frequency':data.app_open_frequency,
        'notification_clicked':data.notification_clicked,
        'notification_dismissed':data.notification_dismissed
        }])

        #model1
        classifcation_model = input_df[[
                          'notification_text',
                          'app_name',
                          'category',
                          'message_length',
                          'contains_urgent_words',
                          'time_of_day',
                          'weekday',
                          'screen_on',
                          'battery_level',
                          'headset_connected',
                          'notification_frequency_today'                          
        ]]

        #one hot encoding of the catagorical data
        eapp = oe_app.transform(classifcation_model[['app_name']])
        classifcation_model = pd.concat([classifcation_model,pd.DataFrame(eapp.toarray(),columns= oe_app.get_feature_names_out(),index=classifcation_model.index)],axis = 1)
        classifcation_model.drop(['app_name'] , axis=1, inplace=True)

        category = oe_category.transform(classifcation_model[['category']])
        classifcation_model = pd.concat([classifcation_model,pd.DataFrame(category.toarray(),columns= oe_category.get_feature_names_out(),index=classifcation_model.index)],axis = 1)
        classifcation_model.drop(['category'] , axis=1, inplace=True)

        time = oe_time.transform(classifcation_model[['time_of_day']])
        classifcation_model = pd.concat([classifcation_model,pd.DataFrame(time.toarray(),columns= oe_time.get_feature_names_out(),index=classifcation_model.index)],axis = 1)
        classifcation_model.drop(['time_of_day'] , axis=1, inplace=True)


        #text processing

        classifcation_model['weekday'] = classifcation_model['weekday'].apply(weekday_and_weekend)
        classifcation_model.loc[:, 'notification_text'] = classifcation_model['notification_text'].apply(demojizeing_notification)
        classifcation_model['notification_text'] = classifcation_model['notification_text'].apply(NLP)


        #vectorization
        vectorized = vector.transform(classifcation_model['notification_text'])
        classifcation_model = pd.concat([classifcation_model,pd.DataFrame(vectorized.toarray(),columns=vector.get_feature_names_out(),index=classifcation_model.index)],axis = 1)
        classifcation_model.drop(['notification_text'], axis = 1, inplace =True)


        #matching the feature order
        classifcation_model = classifcation_model[feature_order_model1]

        #model1 prediction
        model1_priority_label = model1.predict(classifcation_model)[0]

        input_df['model1_priority_label'] = model1_priority_label


        #model2
        regression_model = input_df[[
                           'model1_priority_label',
                           'app_name',
                           'battery_level',
                           'screen_on',
                           'headset_connected',
                           'time_of_day',
                           'weekday',
                           'notification_frequency_today',
                           'app_usage_minutes_today',
                           'app_open_frequency',
                           'notification_clicked', 
                           'notification_dismissed'                         
        ]]
        regression_model['model1_priority_label'] = model1_priority_label

        #preprocesing
        regression_model['weekday']= regression_model['weekday'].apply(weekday_and_weekend)

        #one hot encoding of the catagorical data
        model1_label = oe_model1_priority_label.transform(regression_model[['model1_priority_label']])
        regression_model = pd.concat([regression_model,pd.DataFrame(model1_label.toarray(),columns= oe_model1_priority_label.get_feature_names_out(),index=regression_model.index)],axis = 1)
        regression_model.drop(['model1_priority_label'] , axis=1, inplace=True)

        ecapp = oer_app.transform(regression_model[['app_name']])
        regression_model = pd.concat([regression_model,pd.DataFrame(ecapp.toarray(),columns= oer_app.get_feature_names_out(),index=regression_model.index)],axis = 1)
        regression_model.drop(['app_name'] , axis=1, inplace=True)

   
        etime = oe_time.transform(regression_model[['time_of_day']])
        regression_model = pd.concat([regression_model,pd.DataFrame(etime.toarray(),columns= oe_time.get_feature_names_out(),index=regression_model.index)],axis = 1)
        regression_model.drop(['time_of_day'] , axis=1, inplace=True)
    
        #matching the feature order
        regression_model = regression_model[feature_order_model2]

        #model prediction
        async with model2_lock:
            final_prediction = float(model2.predict(regression_model)[0]
    )

        

        output_content = {"app_name":data.app_name,
                            "notification_text": data.notification_text,
                            "final_prediction":final_prediction}
        


        db_data = {"model1_priority_label": model1_priority_label,
                    "app_name": data.app_name,
                    "battery_level": data.battery_level,
                    "screen_on": data.screen_on,
                    "headset_connected": data.headset_connected,
                    "time_of_day": data.time_of_day,
                    "weekday": data.weekday,
                    "notification_frequency_today": data.notification_frequency_today,
                    "app_usage_minutes_today": data.app_usage_minutes_today,
                    "app_open_frequency": data.app_open_frequency,
                    "notification_clicked": data.notification_clicked,
                    "notification_dismissed": data.notification_dismissed,
                    "priority_score": final_prediction
        }

        await notif_ai_collection.insert_one(db_data)               
        app.state.latest_prediction = output_content
        
        global retraining_in_progress
        count = await notif_ai_collection.count_documents({})
        if count > 0 and count >= 5000 and count % 5000 == 0 and not retraining_in_progress:
            retraining_in_progress =True
            data_from_db = await notif_ai_collection.find().to_list(length=None)
            asyncio.create_task(retrain_model2(data_from_db))
           
    
        return JSONResponse(status_code=200, content=output_content)        
    except Exception as e:
        print("ERROR:", e)
        raise

@app.get('/get_model_data')
def get_model_data():
    
    return JSONResponse(status_code=200, content=app.state.latest_prediction)
    

# database :::::::::::::::


class notification_model2_feature_db(BaseModel):
    model1_priority_label: Annotated[Literal["High", "Medium", "Low"],
                                      Field(description=" high, medium, low ")]
    app_name: Annotated[str,Field(description="the app name of the notification")]
    battery_level:Annotated[int,Field(description="the battery percentage of the phone when the notification arrived")]
    screen_on: Annotated[bool,Field(description="if the screen is on during the notification arrive or not ,yes = 1 and no=0")]
    headset_connected:Annotated[bool,Field(description="Whether the head set was connected when the notification arrived yes =1 and no = 0")]
    time_of_day:Annotated[str,Field(description="Time period when the notification was received (Morning, Afternoon, Evening, or Night).")]
    weekday:Annotated[str,Field(description="the day of the week")]
    notification_frequency_today:Annotated[int , Field(description="Number of notifications received from the same app today.")]
    app_usage_minutes_today:Annotated[int, Field(description="Total minutes the user spent using the app today")]
    app_open_frequency:Annotated[int , Field(description="Number of times the user opened the app today")]
    notification_clicked:Annotated[bool,Field(description="Whether the previous notification from the same app was clicked.")]
    notification_dismissed:Annotated[bool,Field(description="Whether the previous notification from the same app was dismissed.")]
    priority_score:Annotated[float,Field(description="the predicted priority score")]


@app.post('/db_post')
async def data_base_insert(st_data:notification_model2_feature_db):
    result = await notif_ai_collection.insert_one(st_data.model_dump())
    print(result)
    return {"msg":st_data}



async def retrain_model2 (data_from_db):
    global model2
    global retraining_in_progress
    try:
       new_data = pd.DataFrame(data_from_db)
       new_data.drop(['_id'],axis = 1,inplace = True, errors='ignore')
       print("Retraining started")
       #weekdays processing 
       new_data['weekday']= new_data['weekday'].apply(weekday_and_weekend)

       #one hot encoding of the catagorical data
       model1_label = oe_model1_priority_label.transform(new_data[['model1_priority_label']])
       new_data = pd.concat([new_data,pd.DataFrame(model1_label.toarray(),columns= oe_model1_priority_label.get_feature_names_out())],axis = 1)
       new_data.drop(['model1_priority_label'] , axis=1, inplace=True)

       ecapp = oer_app.transform(new_data[['app_name']])
       new_data = pd.concat([new_data,pd.DataFrame(ecapp.toarray(),columns= oer_app.get_feature_names_out())],axis = 1)
       new_data.drop(['app_name'] , axis=1, inplace=True)

   
       etime = oe_time.transform(new_data[['time_of_day']])
       new_data = pd.concat([new_data,pd.DataFrame(etime.toarray(),columns= oe_time.get_feature_names_out())],axis = 1)
       new_data.drop(['time_of_day'] , axis=1, inplace=True)

       #feature matching 
       new_data = new_data[feature_order_model2_for_retrain]

       #feature split
       x = new_data.drop(['priority_score'], axis = 1)
       y = new_data['priority_score']   
       print(f"Training rows: {len(x)}")

       #model 2 retrain
       async with model2_lock:
            model2.fit(x, y)

            with open("model2.pkl", "wb") as f:
                pickle.dump(model2, f)  
       print("Retraining completed")

    except Exception as e:
        print("Retraining failed:", e)
         
    finally:
        retraining_in_progress = False   
    



