import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pillow_avif
import json
import os

# --- PAGE 1: SETUP & DESIGN CONFIG ---
st.set_page_config(
    page_title="VrikshAI: Smart Crop Diagnostic Tool",
    page_icon="🌱",
    layout="wide", # Allows a side-by-side dashboard look
    initial_sidebar_state="expanded"
)

# Professional Header & Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/144/green-leaf--v1.png", width=100)
    st.title("VrikshAI")
    st.markdown("Developed by Team VrikshAI| B.Tech AI & Data Engg")
    
    st.write("---")
    st.markdown("### 🌡️ Field Intake")
    st.caption("Provide current field conditions to improve analysis context.")
    
    # Interactive sliders for user input
    user_temp = st.slider("Current Temperature (°C)", 10, 45, 28)
    user_humidity = st.slider("Current Humidity (%)", 10, 100, 65)

# --- PAGE 2: ASSET LOADING ---
@st.cache_resource # Keeps the model/remedies in memory so it doesn't reload every click
def load_vrikshai_assets():
    """Load model, class names dictionary, and remedies database."""
    # Using the path on your HP Victus
    base_path = '/home/laksh/Storage/Project'
    
    # 1. Load the 87 class names (Now includes ZZZ_Unknown)
    with open(os.path.join(base_path, 'class_names.json'), 'r') as f:
        classes = json.load(f)
    
    # 2. Load the Actionable Advice 
    with open(os.path.join(base_path, 'remedies.json'), 'r') as f:
        remedies = json.load(f)
        
    # 3. Reconstruct & Load Model
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    # Dynamically set to len(classes) so it automatically adjusts to 87
    model.classifier[1] = nn.Linear(num_features, len(classes)) 
    
    # Updated to the new rebranded filename
    model_path = os.path.join(base_path, 'vrikshai_best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu')) 
    model.eval()
    
    return model, classes, remedies

# Safely load assets into variables
try:
    with st.spinner("AI is powering up... Please wait..."):
        # Updated function call to match the new name
        model, class_names, remedies_db = load_vrikshai_assets()
    st.toast("VrikshAI activated successfully!", icon="✅")
except FileNotFoundError as e:
    st.error(f"⚠️ Critical Error: Missing necessary asset file. Please ensure all JSON files are in the Project directory. Error: {e}")
    st.stop()

# --- PAGE 3: IMAGE PREPROCESSING ---
def preprocess_uploaded_image(img):
    """Resizes and normalizes image exactly like our PyTorch training loop."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --- PAGE 4: THE USER INTERFACE LAYOUT ---
st.markdown("# 🌱 VrikshAI Smart Crop Diagnostic Tool")
st.markdown("Upload a clear photo of a crop leaf to identify diseases and receive health analysis.")
st.write("---")

# Main Interface layout: Left column for Upload, Right column for Report
col_upload, col_report = st.columns([2, 3]) 

# Left Side: Image Upload
# Left Side: Image Upload
with col_upload:
    st.subheader("Leaf Upload")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        # EXPANDED FORMATS: This tells the web UI to accept the new files without the red error
        type=["jpg", "jpeg", "png", "webp", "avif", "bmp", "tiff"], 
        help="Use a high-quality photo of a single leaf."
    )

    if uploaded_file is not None:
        # ML SAFETY CATCH: Force conversion to 3-channel RGB immediately to strip transparency
        input_image = Image.open(uploaded_file).convert('RGB')
        
        st.image(input_image, caption='Uploaded Photo (Click to zoom)', use_container_width=True)
    else:
        st.info("👈 Use the button above to upload a photo of a crop leaf.")
        st.markdown(
            """
            <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 2px dashed #cccccc; text-align: center;">
                <h3 style="color: #666666;">Scan Pending</h3>
                <p style="color: #999999;">Waiting for crop photo input...</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Right Side: Diagnosis Report
with col_report:
    st.subheader("📋 Diagnosis Report")

    if uploaded_file is not None:
        with st.spinner("VrikshAI is analyzing the leaf tissue..."):
            
            # Process image and predict
            input_tensor = preprocess_uploaded_image(input_image)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            # Map index to folder name
            raw_result = class_names[predicted_idx]
            conf_score_pct = confidence.item() * 100
            
            # ==========================================
            # THE DOUBLE-LOCK SECURITY GATE
            # ==========================================
            # Lock 1: The Garbage Class (Checks for exact matches)
            if raw_result == "ZZZ-Unknown" or raw_result == "ZZZ_Unknown":
                st.error("### 🛑 Scan Rejected: Non-Crop Image Detected")
                st.write("The AI has determined that this image is not one of our supported agricultural crops.")
                st.info("Please upload a clear, focused image of a supported crop leaf.")
                st.stop() 
            
            # Lock 2: The Confidence Threshold (Catches confusing background bias)
            if conf_score_pct < 70.0:
                st.warning("### ⚠️ Unreliable Scan Detected")
                st.write(f"**Confidence Score: {conf_score_pct:.1f}%**")
                st.write("The AI is not confident enough to provide a safe diagnosis. This usually happens if:")
                st.markdown("- The image is not a supported crop (e.g., weeds, ornamental plants, animals).\n- The leaf is too blurry, far away, or poorly lit.\n- Multiple overlapping diseases are present.")
                st.info("To prevent incorrect chemical application, the treatment report has been withheld. Please try again with a better photo.")
                st.stop()
            # ==========================================

            # Format the output string for the UI
            clean_result = raw_result.replace("___", ": ").replace("_", " ")

            # Look up symptoms/remedies; provide defaults if missing
            remedy_data = remedies_db.get(raw_result, {
                "symptoms": "Please consult a plant pathology specialist.",
                "remedy": "Actionable advice is currently being updated for this disease."
            })
            
            symptoms = remedy_data.get("symptoms", "Data unavailable.")
            remedy = remedy_data.get("remedy", "Please consult a specialist.")
            pref_conditions = remedy_data.get("conditions", "Specific environmental data unavailable for this class.")

        # Display Health Analysis Header
        st.write("### AI Health Analysis")
        
        # Split metrics into two small columns
        c1, c2 = st.columns(2)
        with c1:
            if conf_score_pct > 75:
                st.success(f"### Diagnosis: **{clean_result}**")
            elif conf_score_pct > 50:
                 st.warning(f"### Prediction: {clean_result}")
            else:
                st.error(f"### Unsure: {clean_result}?")
                st.write(f"The AI is only {conf_score_pct:.1f}% confident. This is not a reliable diagnosis.")

        with c2:
            st.metric(label="AI Confidence Score", value=f"{conf_score_pct:.1f}%", delta=None)
            
        st.write("---")
        
        # Display Symptoms
        st.markdown("#### Observed Symptoms")
        st.markdown(
            f"""
            <div style="background-color: #fdfdfd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffa500; color: #333333;">
                <p>{symptoms}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display Remedy
        st.markdown("#### Recommended Actions / Remedies")
        st.markdown(
            f"""
            <div style="background-color: #f9fdf9; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; color: #333333;">
                <p>{remedy}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display Environment Data Profile
        st.write("---")
        st.markdown("#### 🌍 Environmental Risk Analysis")
        st.caption("Comparing your reported field conditions against the pathogen's ideal environment.")
        
        env1, env2 = st.columns(2)
        with env1:
            st.info(
                f"**Your Reported Field Conditions:**\n"
                f"* Temperature: {user_temp}°C\n"
                f"* Humidity: {user_humidity}%"
            )
        with env2:
            st.warning(
                f"**Pathogen's Ideal Environment:**\n"
                f"{pref_conditions}"
            )
        
    else:
        # Default placeholder when no image is uploaded
        st.markdown(
            """
            <div style="background-color: #f9f9f9; padding: 40px; border-radius: 10px; border: 1px solid #eeeeee; text-align: center; color: #aaaaaa;">
                <h3>Diagnosis Report Summary</h3>
                <p>Upload a leaf photo on the left to activate AI analysis and generate a detailed diagnostic report.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- PAGE 5: FOOTER ---
st.write("---")
st.image("https://img.icons8.com/color/48/green-leaf--v1.png", width=25) 
st.caption("© 2026 VrikshAI | Developed by Team Vriksh for B.Tech Final Year")