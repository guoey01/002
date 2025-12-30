import os

import streamlit as st
import pandas as pd
import joblib
import shap

# 输入所有建模参数
vars =[ "Pain" ,"Retire" ,"MCV" ]

# 初始化 session_state 中的 data
# 创建一个空的DataFrame来存储预测数据
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(
        columns=[vars[0], vars[1], vars[2],'Prediction Label' ,'Label'])

# 在主页面上显示数据
st.header('Depression risk of adult patients with arthritis or rheumatism based on LightGBM')

# 创建两列布局
left_column, col1, col2, col3, right_column = st.columns(5)

# 在左侧列中添加其他内容
left_column.write("")

# 在右侧列中显示图像
dirs = os.getcwd()

# 在右侧列中显示图像
right_column.image('./hospital.png', caption='', width=100)

# 创建一个侧边栏
st.sidebar.header('输入参数')

# Input bar 1
a = st.sidebar.number_input(vars[0]+"(0-No,1-Yes)" ,min_value=0 ,max_value=1,value=0)
b = st.sidebar.number_input(vars[1]+"(0-No,1-Yes)" ,min_value=0 ,max_value=1,value=0)
c = st.sidebar.number_input(vars[2]+"(fl)" ,min_value=0.0 ,value=0.0)


# Unpickle classifier
mm = joblib.load('./LightGBM.pkl')

# If button is pressed
if st.sidebar.button("Submit"):
    # Store inputs into dataframe
    X = pd.DataFrame([[a, b, c ]],
                     columns=[vars[0], vars[1], vars[2]])

    # Get prediction
    for index, row in X.iterrows():
        data1 = row.to_frame()
        data2 = pd.DataFrame(data1.values.T, columns=data1.index)
        result111 = mm.predict(data2)
        result222 = str(result111).replace("[", "")
        result = str(result222).replace("]", "")  # 预测结果
        result333 = mm.predict_proba(data2)
        result444 = str(result333).replace("[[", "")
        result555 = str(result444).replace("]]", "")
        strlist = result555.split(' ')
        result_prob_neg = round(float(strlist[0]) * 100, 2)
        if len(strlist[1]) == 0:
            result_prob_pos = 'The conditions do not match and cannot be predicted'
        else:
            # result_prob_pos = round(float(strlist[1]) * 100, 2)  # 预测概率
            result_prob_pos = float(strlist[1])  # 预测概率
    explainer = shap.TreeExplainer(mm)
    shap_values = explainer.shap_values(data2)
    shap_values = shap_values.reshape((1, -1))


    # Output prediction
    st.text(f"The probability of GradientBoost is: {str(result_prob_pos)}%")


    # 创建一个新的DataFrame来存储用户输入的数据
    new_data = pd.DataFrame([[a, b, c , result_prob_pos, None]],
                            columns=st.session_state['data'].columns)

    # 将预测结果添加到新数据中
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 读取 Excel 文件
    df = pd.read_excel(uploaded_file)

    # 列名映射字典,左为Excel字段，右为模型参数名
    column_mapping = {
        vars[0]: vars[0],
        vars[1]: vars[1],
        vars[2]: vars[2],
    }

    # 假设 'Label' 列在 Excel 文件中存在并且不参与计算
    label_column = 'label'  # 这是 Excel 文件中未参与计算的列名

    # 进行列名映射
    df = df.rename(columns=column_mapping)

    # 检查是否所有必需的列都存在
    missing_cols = [col for col in [vars[0], vars[1], vars[2]] if
                    col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        # 逐行读取数据并进行预测
        for _, row in df.iterrows():
            # 提取每一行数据并转换为模型输入格式
            X = pd.DataFrame([row],
                             columns=[vars[0], vars[1], vars[2]])

            # 进行预测
            result = mm.predict(X)[0]
            result_prob = mm.predict_proba(X)[0][1]

            # 获取标签列的值
            label = row[label_column] if label_column in row else None

            # 将结果添加到 session_state 的 data 中
            new_data = pd.DataFrame([[row[vars[0]], row[vars[1]], row[vars[2]] , result_prob, label]],
                                    columns=st.session_state['data'].columns)
            st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 显示更新后的 data
st.write(st.session_state['data'])

# Footer

st.write(
    "<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>",
    unsafe_allow_html=True)
st.markdown('<div style="font-size: 12px; text-align: right;">Powered by MyLab+ i-Research Consulting Team</div>',
            unsafe_allow_html=True )
