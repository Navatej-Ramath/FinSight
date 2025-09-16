import streamlit as st
def render_final_score_gauge(score, max_score):
    """
    Renders a final score gauge and bias based on the normalized score percentage.
    """
    if max_score == 0:
        st.warning("Final score could not be calculated.")
        return

    score_percent = (score / max_score) * 100
    
    if score_percent >= 60: bias = "Strong Bullish Bias"
    elif 30 <= score_percent < 60: bias = "Cautiously Bullish"
    elif -30 < score_percent < 30: bias = "Neutral / Sideways"
    elif -60 < score_percent <= -30: bias = "Cautiously Bearish"
    else: bias = "Strong Bearish Bias"

    if score_percent > 30:
        st.success(f"### Final Verdict: **{bias}**")
    elif score_percent < -30:
        st.error(f"### Final Verdict: **{bias}**")
    else:
        st.warning(f"### Final Verdict: **{bias}**")

    # Normalize score for progress bar (0 to 100)
    normalized_score_for_progress = (score_percent + 100) / 2
    st.progress(int(normalized_score_for_progress), text=f"Overall Score: {score}/{max_score}")

def simple_trend_box(name, trend):
    """Simplified version with fixed styling using your specified colors"""
    if "Bullish" in trend:
        # Matches st.success() green - modern green palette
        bg_color = "#bee7c9"  # Lighter, more modern green
        text_color = "#558967" # Darker, more readable green
        border_color = "#bee7c9"  # Green border
    elif "Bearish" in trend:
        # Matches st.error() red - modern red palette  
        bg_color = "#3e2428"  # Light, soft red
        text_color = "#c18e8c" # Strong, clear red
        border_color = "#3e2428" # Red border
    else:
        bg_color = "#feefc4"  # Lighter, more modern green
        text_color = "#beac79" # Darker, more readable green
        border_color = "#feefc4" # Red border
    
    st.markdown(
        f'<div style="display: inline-block; padding: 8px 16px; '
        f'background-color: {bg_color}; border: 2px solid {border_color}; '
        f'border-radius: 10px; color: {text_color}; font-weight: bold; margin: 5px; '
        f'box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
        f'{name}'
        '</div>',
        unsafe_allow_html=True
    )
def result_message(trend, score,maxscore):
    """
    Creates a styled message with a right-aligned score.
    message_type: 'success' (green) or 'error' (red)
    """
    if (score == 0):
        bg_color = "#feefc4"  # Lighter, more modern green
        text_color = "#beac79" # Darker, more readable green
        border_color = "#feefc4"
    elif score > 0:
    # Matches st.success() green
        bg_color = "#bee7c9"  # Lighter, more modern green
        text_color = "#558967" # Darker, more readable green
        border_color = "#bee7c9" # Google-style green
    else:
    # Matches st.error() red
        bg_color = "#3e2428"  # Light, soft red
        text_color = "#c18e8c" # Strong, clear red
        border_color = "#3e2428" # Google-style red
    
    html = f"""
    <div style="background-color:{bg_color}; color:{text_color}; padding:10px; border-radius:5px; border:0px solid {border_color}; margin: 10px 0px;">
        <div style="display: flex; justify-content: space-between;">
            <span><strong>Analysis:</strong> {trend}</span>
            <span><strong>Score:</strong> {score}/{maxscore}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

