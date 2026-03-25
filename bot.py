#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bot dự đoán Tài/Xỉu (Windows 32/64-bit compatible)
- Tương thích với cả Python 32-bit và 64-bit trên Windows
- /chaybot: bắt đầu gửi dự đoán định kỳ
- /tatbot: dừng gửi dự đoán
- Huấn luyện nhanh LSTM + RF + SVM + XGB khi khởi động
"""

import os
import sys
import time
import random
import warnings
import platform
import numpy as np
from datetime import datetime

# Kiểm tra phiên bản Python và hệ điều hành
print(f"[SYSTEM] Python {platform.python_version()} ({platform.architecture()[0]}) on {platform.system()}")

# Cấu hình tương thích
if sys.maxsize > 2**32:
    print("[SYSTEM] Running 64-bit Python")
else:
    print("[SYSTEM] Running 32-bit Python - Adjusting configurations")
    # Giảm kích thước mô hình cho 32-bit
    os.environ["TF_CPU_ALLOCATOR"] = "cuda_malloc_async"  # Giảm sử dụng bộ nhớ cho TensorFlow

# Telegram
try:
    from telegram import Update
    from telegram.ext import Updater, CommandHandler, CallbackContext
except ImportError:
    print("Thư viện python-telegram-bot chưa được cài đặt. Chạy lệnh sau để cài đặt:")
    print("pip install python-telegram-bot==13.15")
    sys.exit(1)

# Các thư viện ML - xử lý import phù hợp cho 32-bit
try:
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    # XGBoost có thể gặp vấn đề trên 32-bit
    try:
        from xgboost import XGBClassifier
        XGB_ENABLED = True
    except:
        XGB_ENABLED = False
        print("[WARNING] XGBoost không khả dụng trên hệ thống này")

    # TensorFlow/Keras - điều chỉnh cho 32-bit
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.utils import to_categorical
        TF_ENABLED = True
    except Exception as e:
        print(f"[WARNING] TensorFlow không khả dụng: {e}")
        TF_ENABLED = False

except ImportError as e:
    print(f"Thư viện ML chưa được cài đặt. Lỗi: {e}")
    print("Chạy lệnh sau để cài đặt các thư viện cần thiết:")
    print("pip install numpy scikit-learn requests xgboost tensorflow")
    sys.exit(1)

# ===================== CONFIG =====================
BOT_TOKEN = "
8273056780:AAEx2pRI_Tr_cwUN7JoDgDMuc0FW9qoqMRY"  # Thay bằng token thật
ADMIN_ID = 7071414779
CHAT_ID = -1002845092108
MODEL_PATH = "lstm_model.h5"

# Điều chỉnh cho hệ thống 32-bit
if sys.maxsize <= 2**32:
    MODEL_PATH = "lstm_model_32bit.h5"  # Tên file khác để tránh xung đột

API_URL = ""
SEND_INTERVAL = 5  # giây

# ===================== STATE =====================
BOT_RUNNING = False

# ===================== MODEL CREATION =====================
def create_lstm(input_shape):
    """Tạo mô hình LSTM với cấu hình phù hợp cho 32/64-bit"""
    model = Sequential()
    
    # Giảm kích thước mô hình cho 32-bit
    if sys.maxsize <= 2**32:
        model.add(LSTM(32, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(16))
    else:
        model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.25))
        model.add(LSTM(32))
    
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))  # Giảm units cho 32-bit
    model.add(Dense(2, activation="softmax"))
    
    # Sử dụng optimizer tiết kiệm bộ nhớ hơn cho 32-bit
    optimizer = "adam" if sys.maxsize > 2**32 else "rmsprop"
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

def load_lstm_if_exists():
    if os.path.exists(MODEL_PATH):
        try:
            print("[INFO] Đang tải mô hình LSTM từ ổ đĩa...")
            model = load_model(MODEL_PATH)
            model.compile(loss="categorical_crossentropy", 
                        optimizer="adam", 
                        metrics=["accuracy"])
            return model
        except Exception as e:
            print(f"[WARNING] Không thể tải mô hình, sẽ huấn luyện lại: {e}")
    return None

def save_lstm(model):
    try:
        model.save(MODEL_PATH)
        print(f"[INFO] Đã lưu mô hình LSTM: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi lưu mô hình: {e}")

# ===================== DATA HANDLING =====================
def get_api_sample():
    """Lấy dữ liệu mẫu từ API hoặc tạo dữ liệu giả"""
    if not API_URL:
        return None
    
    try:
        res = requests.get(API_URL, timeout=6)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"[ERROR] Lỗi khi lấy dữ liệu từ API: {e}")
    
    return None

def to_feature_vector(api_obj):
    """Chuyển đổi dữ liệu API thành vector đặc trưng"""
    if isinstance(api_obj, dict):
        try:
            # Xử lý dữ liệu API thực tế nếu có
            if "Tong" in api_obj:
                base = int(str(api_obj["Tong"])) if str(api_obj["Tong"]).isdigit() else 0
                return [(base + i) % 2 for i in range(10)]
        except Exception:
            pass
    
    # Trả về dữ liệu giả nếu không có API hoặc xử lý thất bại
    return [random.randint(0, 1) for _ in range(10)]

# ===================== TRAINING =====================
def train_models():
    """Huấn luyện các mô hình với cấu hình phù hợp cho hệ thống 32/64-bit"""
    print("[TRAIN] Đang tạo tập dữ liệu huấn luyện...")
    
    # Tạo dữ liệu huấn luyện
    n_samples = 500 if sys.maxsize <= 2**32 else 1000  # Giảm mẫu cho 32-bit
    X = np.random.randint(0, 2, (n_samples, 10))
    y = np.random.randint(0, 2, n_samples)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    
    # Xử lý cho LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    y_train_cat = to_categorical(y_train, num_classes=2)
    
    # Huấn luyện LSTM nếu có TensorFlow
    lstm = None
    if TF_ENABLED:
        lstm = load_lstm_if_exists()
        if lstm is None:
            lstm = create_lstm((X_train_lstm.shape[1], 1))
            print("[TRAIN] Đang huấn luyện LSTM...")
            epochs = 5 if sys.maxsize <= 2**32 else 8  # Giảm epoch cho 32-bit
            lstm.fit(X_train_lstm, y_train_cat, epochs=epochs, batch_size=16, verbose=0)
            save_lstm(lstm)
        else:
            print("[TRAIN] Sử dụng mô hình LSTM đã tải.")
    else:
        print("[TRAIN] Bỏ qua LSTM do TensorFlow không khả dụng.")
    
    # Huấn luyện các mô hình khác
    print("[TRAIN] Đang huấn luyện RF/SVM/XGB...")
    
    # Random Forest
    rf_params = {
        'n_estimators': 50 if sys.maxsize <= 2**32 else 150,
        'max_depth': 5 if sys.maxsize <= 2**32 else None,
        'random_state': 42
    }
    rf = RandomForestClassifier(**rf_params).fit(X_train, y_train)
    
    # SVM
    svm = SVC(probability=True, kernel="rbf", gamma='scale').fit(X_train, y_train)
    
    # XGBoost (nếu có)
    xgb = None
    if XGB_ENABLED:
        try:
            xgb_params = {
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'n_estimators': 50 if sys.maxsize <= 2**32 else 100,
                'max_depth': 3 if sys.maxsize <= 2**32 else 4,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': 42
            }
            xgb = XGBClassifier(**xgb_params).fit(X_train, y_train)
        except Exception as e:
            print(f"[WARNING] Không thể huấn luyện XGBoost: {e}")
            XGB_ENABLED = False
    
    print("[TRAIN] Hoàn tất huấn luyện.")
    return lstm, rf, svm, xgb, le

# ===================== PREDICTION =====================
def predict_all(lstm, rf, svm, xgb, le, features_10):
    """Dự đoán từ tất cả mô hình và tổng hợp kết quả"""
    X = np.array(features_10, dtype=float).reshape(1, -1)
    predictions = {}
    details = {}
    
    # Dự đoán LSTM nếu có
    lstm_pred = None
    if lstm is not None:
        try:
            X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
            lstm_proba = lstm.predict(X_lstm, verbose=0)[0]
            lstm_pred = int(np.argmax(lstm_proba))
            details["LSTM"] = f"{lstm_pred} ({max(lstm_proba):.2%})"
            predictions["LSTM"] = lstm_pred
        except Exception as e:
            print(f"[ERROR] Lỗi dự đoán LSTM: {e}")
    
    # Dự đoán Random Forest
    rf_proba = rf.predict_proba(X)[0]
    rf_pred = int(np.argmax(rf_proba))
    details["RF"] = f"{rf_pred} ({max(rf_proba):.2%})"
    predictions["RF"] = rf_pred
    
    # Dự đoán SVM
    svm_proba = svm.predict_proba(X)[0]
    svm_pred = int(np.argmax(svm_proba))
    details["SVM"] = f"{svm_pred} ({max(svm_proba):.2%})"
    predictions["SVM"] = svm_pred
    
    # Dự đoán XGBoost nếu có
    xgb_pred = None
    if xgb is not None and XGB_ENABLED:
        try:
            xgb_proba = xgb.predict_proba(X)[0]
            xgb_pred = int(np.argmax(xgb_proba))
            details["XGB"] = f"{xgb_pred} ({max(xgb_proba):.2%})"
            predictions["XGB"] = xgb_pred
        except Exception as e:
            print(f"[ERROR] Lỗi dự đoán XGBoost: {e}")
    
    # Bỏ phiếu quyết định
    votes = list(predictions.values())
    final_pred = max(set(votes), key=votes.count) if votes else random.randint(0, 1)
    details["Final"] = f"{final_pred}"
    
    return final_pred, details

def pretty_label(v):
    """Chuyển đổi nhãn số thành Tài/Xỉu"""
    return "Tài" if int(v) == 1 else "Xỉu"

# ===================== TELEGRAM HANDLERS =====================
def start_cmd(update: Update, context: CallbackContext):
    """Xử lý lệnh /chaybot"""
    if update.effective_user is None or update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ Bạn không có quyền sử dụng lệnh này.")
        return
    
    global BOT_RUNNING
    BOT_RUNNING = True
    update.message.reply_text("✅ Bot đã bắt đầu hoạt động. Đang gửi dự đoán...")

def stop_cmd(update: Update, context: CallbackContext):
    """Xử lý lệnh /tatbot"""
    if update.effective_user is None or update.effective_user.id != ADMIN_ID:
        update.message.reply_text("❌ Bạn không có quyền sử dụng lệnh này.")
        return
    
    global BOT_RUNNING
    BOT_RUNNING = False
    update.message.reply_text("⏹ Bot đã dừng hoạt động.")

def main_loop(context: CallbackContext):
    """Hàm chính gửi dự đoán định kỳ"""
    global BOT_RUNNING
    if not BOT_RUNNING:
        return
    
    try:
        # Lấy dữ liệu
        api_obj = get_api_sample()
        features = to_feature_vector(api_obj) if api_obj is not None else [random.randint(0, 1) for _ in range(10)]
        
        # Dự đoán
        pred, details = predict_all(
            context.bot_data.get("lstm"),
            context.bot_data["rf"],
            context.bot_data["svm"],
            context.bot_data.get("xgb"),
            context.bot_data["le"],
            features
        )
        
        # Tạo thông báo
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "🎯 <b>DỰ ĐOÁN TÀI/XỈU</b>",
            f"🕒 Thời gian: {ts}",
            f"🧩 Đặc trưng: {features}",
            "",
            "<b>CHI TIẾT MÔ HÌNH:</b>",
            *[f"• {k}: {v}" for k, v in details.items() if k != "Final"],
            "",
            f"🔥 <b>KẾT LUẬN: {pretty_label(details['Final'])}</b>"
        ]
        
        # Gửi tin nhắn
        context.bot.send_message(
            chat_id=ADMIN_ID,
            text="\n".join(lines),
            parse_mode="HTML",
            disable_web_page_preview=True
        )
    except Exception as e:
        print(f"[ERROR] Lỗi trong quá trình dự đoán: {e}")

# ===================== MAIN =====================
if __name__ == "__main__":
    print("[BOT] Đang khởi động...")
    
    # Kiểm tra và cài đặt các thư viện cần thiết
    required_libs = ["numpy", "scikit-learn", "requests"]
    if XGB_ENABLED:
        required_libs.append("xgboost")
    if TF_ENABLED:
        required_libs.append("tensorflow")
    
    print(f"[BOT] Sử dụng các thư viện: {', '.join(required_libs)}")
    
    # Huấn luyện mô hình
    print("[BOT] Đang huấn luyện các mô hình...")
    lstm, rf, svm, xgb, le = train_models()
    
    # Khởi tạo Telegram bot
    if not BOT_TOKEN or BOT_TOKEN.startswith("PASTE_"):
        print("❌ Vui lòng cung cấp BOT_TOKEN hợp lệ")
        sys.exit(1)
    
    try:
        updater = Updater(BOT_TOKEN, use_context=True)
        dp = updater.dispatcher
        
        # Lưu các mô hình vào bot_data để sử dụng sau này
        dp.bot_data["lstm"] = lstm
        dp.bot_data["rf"] = rf
        dp.bot_data["svm"] = svm
        dp.bot_data["xgb"] = xgb
        dp.bot_data["le"] = le
        
        # Đăng ký các lệnh
        dp.add_handler(CommandHandler("chaybot", start_cmd))
        dp.add_handler(CommandHandler("tatbot", stop_cmd))
        
        # Thiết lập job định kỳ
        updater.job_queue.run_repeating(main_loop, interval=SEND_INTERVAL, first=0)
        
        print("[BOT] Đã sẵn sàng. Gửi /chaybot trong Telegram để bắt đầu.")
        updater.start_polling()
        updater.idle()
        
    except Exception as e:
        print(f"❌ Lỗi khi khởi động bot: {e}")
        sys.exit(1)