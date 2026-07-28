import json
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PSEi 30 Quant Intelligence", page_icon="📈", layout="wide"
)

# Read index.html and latest_recommendations.json
with open("index.html", "r", encoding="utf-8") as f:
  html_code = f.read()

# Inject the JSON directly into HTML to bypass iframe fetch security limits
with open("latest_recommendations.json", "r", encoding="utf-8") as f:
  json_data = f.read()

# Replace fetch script with embedded data
script_override = f"<script>const embeddedData = {json_data};</script>"
full_html = script_override + html_code.replace(
    "await fetch('latest_recommendations.json')", "embeddedData"
).replace("await response.json()", "embeddedData")

# Render full screen
components.html(full_html, height=1600, scrolling=True)
