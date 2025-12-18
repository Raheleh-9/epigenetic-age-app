import streamlit as st
import pandas as pd
import numpy as np
import pickle

# تنظیمات ظاهر برنامه
st.set_page_config(page_title="Epigenetic Age Predictor", page_icon="🧬")

st.title("🧬 حرفه‌ای‌ترین پیش‌بینی‌گر سن اپی‌ژنتیک")
st.write("این مدل بر اساس داده‌های واقعی GSE40279 و ۵۰۰ شاخص برتر CpG آموزش دیده است.")

# ۱. بارگذاری مدل و اسکیلر و لیست ویژگی‌ها
@st.cache_resource
def load_model():
    with open("trained_model.pkl", "rb") as f:
        # فایل جدید ما ۳ تا بخش داره: مدل، اسکیلر و نام ستون‌ها
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    st.success("✅ مدل هوشمند با موفقیت بارگذاری شد")
except Exception as e:
    st.error(f"خطا در بارگذاری مدل: {e}")

# ۲. بخش آپلود فایل توسط کاربر
uploaded_file = st.file_uploader("فایل متیلاسیون خود را آپلود کنید (CSV یا TXT)", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # خوندن فایل کاربر
        user_data = pd.read_csv(uploaded_file, index_col=0)
        
        st.info("در حال تطبیق ویژگی‌های فایل شما با مدل مرجع...")

        # ۳. بخش حیاتی: فیلتر کردن و مرتب‌سازی ستون‌ها
        # چک می‌کنیم کدوم یکی از اون ۵۰۰ تا ستون توی فایل کاربر هست
        available_features = [f for f in feature_names if f in user_data.columns]
        missing_features = [f for f in feature_names if f not in user_data.columns]

        if len(available_features) < len(feature_names) * 0.8:
            st.warning(f"⚠️ توجه: فایل شما فقط {len(available_features)} مورد از ۵۰۰ شاخص لازم را دارد. دقت ممکن است کاهش یابد.")
        
        # پر کردن ستون‌های غایب با عدد صفر (یا میانگین) و مرتب‌سازی دقیق ستون‌ها
        input_df = user_data.reindex(columns=feature_names, fill_value=0)

        # ۴. پیش‌پردازش و پیش‌بینی
        # استانداردسازی داده‌ها با اسکیلر مدل اصلی
        input_scaled = scaler.transform(input_df)
        
        # حدس زدن سن
        prediction = model.predict(input_scaled)

        # ۵. نمایش نتیجه با کلاس جهانی
        st.balloons()
        st.subheader("نتایج تحلیل بیولوژیکی:")
        cols = st.columns(2)
        cols[0].metric("سن اپی‌ژنتیک تخمینی", f"{prediction[0]:.1f} سال")
        cols[1].metric("تعداد شاخص‌های تحلیل شده", f"{len(available_features)} CpG")

        st.progress(min(int(prediction[0]), 100))
        st.write("---")
        st.caption("ارائه شده برای تیم تحقیقاتی مایکل لاستگارتن - توسعه یافته با داده‌های واقعی NCBI")

    except Exception as e:
        st.error(f"خطا در پردازش فایل: {e}")
