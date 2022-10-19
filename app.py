#from importlib import reload
#import tensorflow as tf
import streamlit as st
import pandas as pd
#import tensorflow as tf
#import tensorflow_hub as hub
#import tensorflow_text as text
#from bert import inference
from models import inference, inference_st_hyperbole, inference_st_metaphor

#from tensorflow_addons.optimizers import adamw
#tf.keras.optimizers.adamw = adamw

#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(0, '/tf/models/')

import os
#os.environ['PYTHONPATH'] += "/tf/models/"

#import sys
#sys.path.append("/tf/models/")
#export PYTHONPATH=$PYTHONPATH:/tf/models/
#from official.nlp import optimization
#import os
#from eval import evaluate

#export PYTHONPATH='tf/models/'

curr_dir = os.getcwd()
models_dir = curr_dir + "/models/"
st.set_page_config(layout="wide")
st.title("Multi-task Hyperbole and Metaphor Detection")

def my_widget(key):
    return st.button("Submit ")

#option = st.selectbox(
#     'Select the type of task',
#     ('Single Sentence', 'Two Sentence'))

#st.write('You selected:', option)


with st.form(key='my_form'):
    
    #print(option)
    
    #if option == 'Single Sentence':

    text_area_input1 = st.text_area("Enter Sentence:")

    #else:
    #    text_input2 = st.text_input("Enter Sentence-1:")
    #    text_input3 = st.text_input("Enter Sentence-2:")


    submitButton = st.form_submit_button(label = 'Predict')




if submitButton:
    #print(text_area_input)
    #print(text_input)
    #examples = []
    #if option == 'Single Sentence':
    #examples.append(text_area_input1)
   # else:
   #     examples.append(text_input2)
   #     examples.append(text_input3)
    #init_lr = 3e-5
    #optimizer = optimization.create_optimizer(init_lr=3e-5,
                                          #num_train_steps=105,
                                          #num_warmup_steps=10,
    #                                      optimizer_type='adamw')
    #reloaded_model = tf.keras.models.load_model(models_dir + 'hypo_red_trained_bert_cased_e3.h5',  custom_objects = {'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer})
    #reloaded_model = tf.keras.models.load_model(models_dir + 'hypo_red_trained_bert_cased_e3.h5', custom_objects = {'KerasLayer': hub.KerasLayer, 'AdamWeightDecay': optimizer})
    #results = tf.sigmoid(reloaded_model(tf.constant(examples)))
    st_hyp_results = inference_st_hyperbole(str(text_area_input1))
    st_met_results = inference_st_metaphor(str(text_area_input1))
    mt_results = inference(str(text_area_input1))

    #hyperbol-st results
    hyp_st_final_list = st_hyp_results.tolist()
    hyp_st_hyperbole_score = round(hyp_st_final_list[0][1], 3)
    #hyp_st_metaphor_score = round(hyp_st_final_list[0][1], 3)
    #answer = "HI"

    #metaphor-st results
    met_st_final_list = st_met_results.tolist()
    #met_st_hyperbole_score = round(met_st_final_list[0][0], 3)
    met_st_metaphor_score = round(met_st_final_list[0][1], 3)

    
    #MT results
    mt_final_list = mt_results.tolist()
    mt_hyperbole_score = round(mt_final_list[0][0], 3)
    mt_metaphor_score = round(met_st_final_list[0][1], 3)



    st_hyperbole_df = pd.DataFrame({'Hyperbolic-score':[hyp_st_hyperbole_score], 'Metaphoric-Score': ['---'], 'Final Result': []}, index=[])
    st_metaphor_df = pd.DataFrame({'Hyperbolic-score':['---'], 'Metaphoric-Score': [met_st_metaphor_score], 'Final Result':[]}, index=[])
    mt_df =  pd.DataFrame({'Hyperbolic-score':[mt_hyperbole_score], 'Metaphoric-Score': [mt_metaphor_score], 'Final Result':[]}, index=[])

    frames = [st_hyperbole_df, st_metaphor_df, mt_df]
    result = pd.concat(frames)
    #print(result)
    st.write("Results:")
    st.markdown(
       f"""
       * Sentence : {str(text_area_input1)}"""
       )
    st.table(result)


    #st.markdown(
    #   f"""
    #   *output : {final_list}
    #   * Sentence : {str(text_area_input1)}
    #   * Hyperbolic Score : {str(hyperbole_score)}
    #   * Metaphoric Score : {str(hyperbole_score)}
    #    """
    




    
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	                content:'Note: This Demo application was made as part of MTP Stage 1 Presentation under the Guidance of Prof. Pushpak Bhattacharyya'; 
	                visibility: visible;
	                display: block;
	                position: relative;
	                #background-color: red;
	                padding: 5px;
	                top: 2px;
                    }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)