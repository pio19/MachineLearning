import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Đã thêm import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Hàm đọc dữ liệu từ tệp CSV với xử lý mã hóa
def load_data(filepath, encoding):
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        df = df.dropna(subset=['Text', 'Rating'])
        df['Rating'] = df['Rating'].astype(int)
        return df
    except UnicodeDecodeError:
        st.error(f"Không thể đọc tệp với mã hóa {encoding}.")
        return None

# Chuyển đổi giá trị Rating thành nhãn phân loại
def convert_rating_to_label(rating):
    if rating in [1, 2]:
        return 'Tiêu cực'
    elif rating == 3:
        return 'Trung lập'
    elif rating in [4, 5]:
        return 'Tích cực'
    else:
        return None

# Huấn luyện mô hình
def train_model(df):
    df['Sentiment'] = df['Rating'].apply(convert_rating_to_label)
    df = df.dropna(subset=['Sentiment'])  # Loại bỏ các hàng không hợp lệ
    X = df['Text']
    y = df['Sentiment']
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Tiêu cực', 'Trung lập', 'Tích cực'])
    return vectorizer, model, accuracy, report

# Dự đoán đánh giá
def predict_sentiment(review, vectorizer, model):
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return prediction[0]

# Giao diện Streamlit
st.title('Phân loại Đánh giá sản phẩm')
st.write('Ứng dụng này giúp phân loại các đánh giá sản phẩm thành Tiêu cực, Trung lập hoặc Tích cực dựa trên nội dung của đánh giá.')

# Tùy chọn mã hóa
encoding_options = ['utf-8', 'latin1', 'cp1252']
encoding_choice = st.selectbox('Chọn mã hóa cho tệp dữ liệu:', encoding_options)

# Đường dẫn đến tệp dữ liệu
data_path = st.text_input('Đường dẫn đến tệp dữ liệu:', r"D:\TaiLieu\MachineLearning\Dataset\data.csv")

if st.button('Tải dữ liệu và huấn luyện mô hình'):
    df = load_data(data_path, encoding_choice)
    if df is not None:
        vectorizer, model, accuracy, report = train_model(df)
        st.session_state['vectorizer'] = vectorizer
        st.session_state['model'] = model
        st.session_state['accuracy'] = accuracy
        st.session_state['report'] = report
        st.success(f'Mô hình đã được huấn luyện với độ chính xác: {accuracy:.2f}')
        st.write('Báo cáo phân loại:')
        st.text(report)
    else:
        st.error('Không thể tải dữ liệu. Vui lòng kiểm tra đường dẫn và mã hóa.')

# Nhập đánh giá sản phẩm để dự đoán
st.subheader('Dự đoán đánh giá')
review_text = st.text_area('Nhập đánh giá sản phẩm:')

if st.button('Dự đoán'):
    if 'model' in st.session_state:
        sentiment = predict_sentiment(review_text, st.session_state['vectorizer'], st.session_state['model'])
        if sentiment == 'Tiêu cực':
            st.markdown(f'### Đánh giá được phân loại là: <span style="color:red">{sentiment}</span>', unsafe_allow_html=True)
        elif sentiment == 'Trung lập':
            st.markdown(f'### Đánh giá được phân loại là: <span style="color:orange">{sentiment}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'### Đánh giá được phân loại là: <span style="color:green">{sentiment}</span>', unsafe_allow_html=True)
    else:
        st.error('Mô hình chưa được huấn luyện. Vui lòng tải dữ liệu và huấn luyện mô hình trước.')
