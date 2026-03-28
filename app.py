import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load CSS from file
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("assets/style.css")

st.markdown("""
    <style>
    .stApp, .stApp *:not(input):not(textarea):not([contenteditable='true']) {
        cursor: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'><defs><radialGradient id='g' cx='50%25' cy='35%25' r='60%25'><stop offset='0%25' stop-color='%23ffd36b'/><stop offset='100%25' stop-color='%23b8860b'/></radialGradient></defs><circle cx='12' cy='12' r='10' fill='url(%23g)' stroke='%230a0a0a' stroke-width='1'/><text x='12' y='16' text-anchor='middle' font-size='12' font-weight='700' fill='%230a0a0a'>%E2%82%B9</text></svg>") 12 12, auto !important;
    }
    </style>
""", unsafe_allow_html=True)

components.html("""
<!DOCTYPE html><html><head><meta charset='utf-8'></head><body></body>
<script>
(function(){
  var d;try{d=(window.parent&&window.parent.document)?window.parent.document:document}catch(e){d=document}
  if(d.getElementById('mc-wheel-fix')) return;
  var f=d.createElement('div');f.id='mc-wheel-fix';f.style.display='none';d.body.appendChild(f);
  function scEl(){return d.scrollingElement||d.documentElement||d.body}
  function onWheel(e){
    if(e.ctrlKey) return;
    e.preventDefault();
    scEl().scrollTop += e.deltaY;
  }
  function bind(){
    var nodes=d.querySelectorAll('input,[role="combobox"],select');
    nodes.forEach(function(el){
      if(el.__mcWheel) return;
      el.addEventListener('wheel', onWheel, {passive:false});
      el.__mcWheel=true;
    });
  }
  bind();
  var MO=(d.defaultView||window).MutationObserver;
  if(MO){ new MO(bind).observe(d.body,{childList:true,subtree:true}); }
})();
</script>
""", height=0, width=0)

components.html("""
<!DOCTYPE html><html><head><meta charset='utf-8'></head><body></body>
<script>
(function(){
  var d;try{d=(window.parent&&window.parent.document)?window.parent.document:document}catch(e){d=document}
  if(d.getElementById('mc-mini-sparkle-style')) return;
  var css='#mc-mini-sparkles{position:fixed;left:0;top:0;width:100vw;height:100vh;pointer-events:none;z-index:2147483644}'+
          '.mc-mini-s{position:absolute;font-size:10px;line-height:1;opacity:.95;color:#ffd36b;text-shadow:0 0 6px rgba(255,211,107,.8),0 0 14px rgba(255,193,7,.5)}'+
          '.mc-mini-dot{width:6px;height:6px;border-radius:50%;background:radial-gradient(circle,#ffd36b 0%,rgba(255,193,7,.6) 45%,transparent 70%);box-shadow:0 0 8px rgba(255,193,7,.6)}'+
          '@keyframes mcFadeRise{0%{transform:translate3d(0,0,0) scale(1);opacity:.95}100%{transform:translate3d(var(--dx),var(--dy),0) scale(.85);opacity:0}}';
  var st=d.createElement('style');st.id='mc-mini-sparkle-style';st.textContent=css;d.head.appendChild(st);
  var layer=d.getElementById('mc-mini-sparkles')||d.body.appendChild(Object.assign(d.createElement('div'),{id:'mc-mini-sparkles'}));
  var last=0, parts=[];
  function spawn(x,y,vx,vy){
    var isIcon=Math.random()<0.35;
    var el=d.createElement(isIcon?'div':'div');
    if(isIcon){el.className='mc-mini-s';el.innerHTML='&#8377;';}
    else{el.className='mc-mini-dot';}
    el.style.left=x+'px';el.style.top=y+'px';
    el.style.position='absolute';
    var dx=((-vx*6)+(Math.random()-0.5)*10).toFixed(1)+'px';
    var dy=((-vy*4)-6+(Math.random()-0.5)*6).toFixed(1)+'px';
    el.style.setProperty('--dx',dx);
    el.style.setProperty('--dy',dy);
    el.style.animation='mcFadeRise 600ms ease-out forwards';
    layer.appendChild(el);
    parts.push(el);
    if(parts.length>32){var r=parts.shift();if(r&&r.parentNode){r.parentNode.removeChild(r);}}
    el.addEventListener('animationend',function(){if(el&&el.parentNode){el.parentNode.removeChild(el);}});
  }
  var pmx=0,pmy=0;
  d.addEventListener('mousemove',function(e){
    var now=performance.now(); if(now-last<40) return; last=now;
    var vx=e.clientX-pmx, vy=e.clientY-pmy; pmx=e.clientX; pmy=e.clientY;
    var sp=Math.min(1, Math.hypot(vx,vy)/30);
    if(sp<0.15) return;
    var n= sp>0.6?2:1;
    for(var i=0;i<n;i++){spawn(e.clientX, e.clientY, vx, vy);}
  });
})();
</script>
""", height=0, width=0)

# Load models
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('models/rf_salary_predictor.pkl')
        clf_model = joblib.load('models/dt_salary_classifier.pkl')
        kmeans_model = joblib.load('models/kmeans_segmentation.pkl')
        cluster_scaler = joblib.load('models/cluster_scaler.pkl')
        return reg_model, clf_model, kmeans_model, cluster_scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def main():
    reg_model, clf_model, kmeans_model, cluster_scaler = load_models()

    if reg_model is None:
        st.error("Failed to load models. Please ensure the models are trained and saved.")
        return

    # Header
    st.markdown('<h1 class="main-header">Employee Salary Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Modern ML-driven salary insights for the Indian Market</p>', unsafe_allow_html=True)

    if 'first_load' not in st.session_state:
        st.session_state['age'] = None
        st.session_state['experience'] = None
        st.session_state['education'] = None
        st.session_state['job_role'] = None
        st.session_state['location'] = None
        st.session_state['skills'] = None
        st.session_state['show_graph'] = None
        st.session_state['first_load'] = True
    if 'show_graph' not in st.session_state:
        st.session_state['show_graph'] = None

    # Input Section
    st.markdown('<h3 class="section-title">Employee Details</h3>', unsafe_allow_html=True)
    
    age = st.number_input("Age", min_value=18, max_value=65, value=st.session_state.get('age', None), key='age', placeholder="Enter age")
    experience = st.number_input("Years of Experience", min_value=0, max_value=45, value=st.session_state.get('experience', None), key='experience', placeholder="Enter years")
    
    education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"], index=None, key='education', placeholder="Select Education")
    job_role = st.selectbox("Job Role", ["Developer", "Data Scientist", "Manager", "Director"], index=None, key='job_role', placeholder="Select Role")
    location = st.selectbox("Location", ["New York", "San Francisco", "Austin", "Remote"], index=None, key='location', placeholder="Select Location")
    skills = st.selectbox("Primary Skill", ["Python", "Java", "SQL", "C++", "AWS"], index=None, key='skills', placeholder="Select Skill")
    
    st.markdown('<div style="text-align: center; margin-top: 2rem;">', unsafe_allow_html=True)
    predict_button = st.button("Predict Salary", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Results Section
    if predict_button:
        if any(v is None for v in [age, experience, education, job_role, location, skills]):
            st.warning("Please fill in all details before predicting.")
        else:
            with st.spinner("Analyzing profile..."):
                try:
                    input_data = pd.DataFrame({
                        'Age': [age], 'Experience': [experience], 'Education': [education],
                        'Job Role': [job_role], 'Location': [location], 'Skills': [skills]
                    })
                    
                    salary_pred = reg_model.predict(input_data)[0]
                    category_pred = clf_model.predict(input_data)[0]
                    
                    clust_input = pd.DataFrame({'Experience': [experience], 'Salary': [salary_pred]})
                    clust_scaled = cluster_scaler.transform(clust_input)
                    cluster_pred = kmeans_model.predict(clust_scaled)[0]
                    
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-title">Prediction Results</div>
                      <div class="metric-row">
                        <div class="metric-card">
                          <div class="metric-label">Predicted Salary (Annual)</div>
                          <div class="metric-value salary-glow">₹{salary_pred:,.0f}</div>
                          <div style="margin-top:6px;color:#94a3b8;font-size:0.9rem;">≈ ₹{salary_pred/12:,.0f} per month</div>
                        </div>
                        <div class="metric-card">
                          <div class="metric-label">Category</div>
                          <div class="metric-value category-glow">{category_pred}</div>
                        </div>
                        <div class="metric-card">
                          <div class="metric-label">Segment</div>
                          <div class="metric-value cluster-glow">Cluster {cluster_pred}</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("What these results mean"):
                        st.markdown("""
                        - Predicted Salary: Estimated annual base salary in INR from a regression model trained on Age, Experience, Education, Job Role, Location, and Skills. It does not include bonuses or stock and is an approximation, not a guarantee.
                        - Category: Salary bracket derived from the predicted salary using fixed thresholds: Low < ₹70,000; Medium ₹70,000–₹1,20,000; High > ₹1,20,000. This helps summarize the range at a glance.
                        - Segment: An unsupervised cluster label based on experience and predicted salary (after scaling). It groups similar profiles together, e.g., early‑career, mid‑level, or senior/high‑earning cohorts. Cluster numbers are identifiers, not rankings.
                        """)
                except Exception as e:
                    st.error(f"Error: {e}")

    # Graphs Section
    st.markdown('<h3 style="text-align: center; color: white; margin-top: 3rem; margin-bottom: 2rem;">Data Visualizations</h3>', unsafe_allow_html=True)
    
    graph_map = {
        'salary_distribution': ('Salary Distribution', 'outputs/salary_distribution.png'),
        'correlation_heatmap': ('Correlation Heatmap', 'outputs/correlation_heatmap.png'),
        'feature_importance': ('Feature Importance', 'outputs/feature_importance.png'),
        'confusion_matrix': ('Confusion Matrix', 'outputs/confusion_matrix.png'),
        'kmeans_elbow': ('Elbow Method', 'outputs/kmeans_elbow_plot.png'),
        'cluster_scatter': ('Cluster Scatter', 'outputs/cluster_scatter.png')
    }

    if st.session_state['show_graph']:
        if st.button("Hide Graph", use_container_width=True):
            st.session_state['show_graph'] = None
            st.rerun()
        
        selected = st.session_state['show_graph']
        label, path = graph_map[selected]
        st.markdown(f"<h4 style='text-align: center; color: #38bdf8; margin-top: 2rem;'>{label}</h4>", unsafe_allow_html=True)
        try:
            st.image(path, use_container_width=True)
        except:
            st.error("Graph image not found.")
    else:
        keys = list(graph_map.keys())
        for row_start in range(0, len(keys), 3):
            cols = st.columns(3)
            for col_idx in range(3):
                i = row_start + col_idx
                if i < len(keys):
                    k = keys[i]
                    label, _ = graph_map[k]
                    with cols[col_idx]:
                        if st.button(label, use_container_width=True, key=f"btn_{k}"):
                            st.session_state['show_graph'] = k
                            st.rerun()

if __name__ == "__main__":
    main()
