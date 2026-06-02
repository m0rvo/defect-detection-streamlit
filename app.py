import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

#  НАСТРОЙКИ СТРАНИЦЫ 
st.set_page_config(page_title="Детекция дефектов", page_icon="🔍", layout="centered")

st.title("🔍 Детекция дефектов литья")
st.markdown("Загрузи фото детали — модель определит: **OK** или **DEFECT**")

#  ЗАГРУЗКА МОДЕЛИ 
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#  ЗАГРУЗКА ФОТО 
uploaded_file = st.file_uploader(" Загрузи изображение детали", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Загруженное изображение", use_container_width=True)

    with col2:
        st.subheader(" Результат")

        # Предсказание
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            conf_ok = probabilities[0][0].item() * 100
            conf_def = probabilities[0][1].item() * 100

        confidence_gap = conf_ok - conf_def

        if conf_ok > conf_def and confidence_gap > 20:
            st.success(f"✅ **OK** — Деталь нормальная")
            st.progress(int(conf_ok))
            st.metric("Уверенность OK", f"{conf_ok:.1f}%")
            st.metric("Уверенность DEFECT", f"{conf_def:.1f}%")
        else:
            st.error(f"❌ **DEFECT** — Обнаружен дефект!")
            st.progress(int(conf_def))
            st.metric("Уверенность DEFECT", f"{conf_def:.1f}%")
            st.metric("Уверенность OK", f"{conf_ok:.1f}%")

        # Кнопка скачать результат
        st.download_button(
            " Скачать результат",
            data=f"Результат: {'OK' if conf_ok > conf_def else 'DEFECT'}\nУверенность: {max(conf_ok, conf_def):.1f}%",
            file_name="result.txt"
        )

else:
    st.info("Загрузи изображение для проверки")

#  ИНФО 
st.markdown("---")
st.markdown("""
### ! Информация
- **Модель**: ResNet18
- **Обучено на**: ~1900 изображений (OK + DEFECT)
- **Эпох**: 10
- **Балансировка классов**:  Включена
""")
