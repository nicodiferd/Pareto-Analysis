"""
Flora Chat Analysis - Streamlit Dashboard
Industrial Engineering Pareto Analysis of Flora User Interactions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import re
from datetime import datetime, timedelta
import extra_streamlit_components as stx

# Page config
st.set_page_config(
    page_title="Flora Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# NAVIGATION
# ============================================================================
PAGES = {
    "Flora Pareto Analysis": "pareto",
    "Top 4 User Deep Dive": "users"
}

# ============================================================================
# AUTHENTICATION (with 15-minute cookie cache)
# ============================================================================
AUTH_CODE = "BLOOMFLORA2026"
AUTH_COOKIE_NAME = "flora_auth"
AUTH_DURATION_MINUTES = 15

# Initialize cookie manager (cannot use @st.cache_resource with widget components)
cookie_manager = stx.CookieManager()

def check_auth():
    """Check if user is authenticated via cookie."""
    auth_cookie = cookie_manager.get(AUTH_COOKIE_NAME)
    if auth_cookie:
        try:
            expiry = datetime.fromisoformat(auth_cookie)
            if datetime.now() < expiry:
                return True
        except (ValueError, TypeError):
            pass
    return False

def set_auth_cookie():
    """Set authentication cookie with 15-minute expiry."""
    expiry = datetime.now() + timedelta(minutes=AUTH_DURATION_MINUTES)
    cookie_manager.set(AUTH_COOKIE_NAME, expiry.isoformat(), expires_at=expiry)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = check_auth()

if not st.session_state.authenticated:
    st.markdown("")
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Flora Pareto Analysis")
        st.markdown("*Enter access code to continue*")
        st.markdown("")
        code = st.text_input("Access Code", type="password", placeholder="Enter code...")
        if st.button("Submit", use_container_width=True):
            if code == AUTH_CODE:
                st.session_state.authenticated = True
                set_auth_cookie()
                st.rerun()
            else:
                st.error("Invalid access code")
    st.stop()

# Data paths
DATA_DIR = Path("data")

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.title("📊 Flora Analysis")
    st.markdown("---")
    selected_page = st.radio(
        "Select Analysis",
        list(PAGES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("BLOOM-02 | Bloomfilter AI")

# ============================================================================
# DATA LOADING & CATEGORIZATION
# ============================================================================

@st.cache_data
def load_data():
    """Load cleaned chat data."""
    df = pd.read_csv(DATA_DIR / "flora-chats-cleaned.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

@st.cache_data
def load_user_analysis():
    """Load top 4 user analysis data."""
    try:
        with open(DATA_DIR / "top4_user_analysis.json", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def categorize_intent(instruction):
    """
    Categorize user intent based on action verbs and targets.
    Returns: (category, confidence, reasoning)
    """
    if pd.isna(instruction) or str(instruction).strip() == '':
        return ('Empty Input', 'high', 'No instruction provided')

    text = str(instruction).lower().strip()

    # Pattern 1: Executive/Summary Requests
    if re.search(r'(provide|give|generate|create).*(executive|summary|overview|report)', text):
        return ('Executive Reporting', 'high', 'Requests summary/executive report')

    # Pattern 2: Data Analysis Requests (with JSON payload)
    if re.search(r'analyze.*(following|this|the).*data', text):
        return ('Data Analysis', 'high', 'Requests analysis of provided data')

    # Pattern 3: Sprint Retrospective/Analysis
    if re.search(r'(sprint|retro|retrospective).*(analysis|review)', text):
        return ('Sprint Retrospective', 'high', 'Requests sprint retrospective analysis')

    # Pattern 4: Velocity/Metrics Queries
    if re.search(r'(velocity|throughput|burndown|metrics)', text):
        return ('Metrics Query', 'high', 'Requests specific metrics')

    # Pattern 5: Information Requests (conversational)
    if re.search(r'^(tell me|what is|what are|describe|explain|show me)', text):
        return ('Information Request', 'high', 'Asks for information/explanation')

    # Pattern 6: Performance Analysis
    if re.search(r'performance.*(analysis|review|check)', text):
        return ('Performance Analysis', 'medium', 'Requests performance analysis')

    # Pattern 7: Work Period / Sprint Context Queries
    if re.search(r'(for|about).*(work.?period|sprint|\\[ref\\])', text):
        if re.search(r'(summary|report|overview)', text):
            return ('Sprint Reporting', 'medium', 'Requests report for specific sprint')
        else:
            return ('Sprint Query', 'medium', 'Query about specific sprint')

    # Pattern 8: Initiative/Project Queries
    if re.search(r'(initiative|project|epic)', text):
        return ('Initiative Query', 'medium', 'Query about initiative/project')

    # Pattern 9: Generic Analysis Request
    if re.search(r'^analyze', text):
        return ('General Analysis', 'medium', 'Generic analysis request')

    return ('Unclassified', 'low', 'No pattern matched')


@st.cache_data
def categorize_all(df):
    """Apply categorization to entire dataframe."""
    categorization = df['instruction'].apply(categorize_intent)
    df = df.copy()
    df['intent_category'] = categorization.apply(lambda x: x[0])
    df['category_confidence'] = categorization.apply(lambda x: x[1])
    return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_pareto_chart(data, title, color='#2E86AB', vital_color='#E63946'):
    """Create a professional Pareto chart with vital few highlighted."""
    # Sort data in descending order
    data = data.sort_values(ascending=False)

    total = data.sum()
    cumulative_pct = data.cumsum() / total * 100
    threshold_idx = (cumulative_pct <= 80).sum()

    # Color bars - vital few in red, trivial many in blue
    colors = [vital_color if i < threshold_idx else color for i in range(len(data))]

    fig = go.Figure()

    # Bar chart - cleaner labels (count only, percentage in hover)
    fig.add_trace(go.Bar(
        x=list(range(len(data))),  # Use numeric x to preserve order
        y=data.values.tolist(),
        name='Count',
        marker_color=colors,
        opacity=0.9,
        text=[str(v) for v in data.values],
        textposition='outside',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{customdata[0]}</b><br>Count: %{y}<br>Percentage: %{customdata[1]:.1f}%<extra></extra>',
        customdata=[[cat, v/total*100] for cat, v in zip(data.index, data.values)]
    ))

    # Cumulative line
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),  # Use numeric x to preserve order
        y=cumulative_pct.values.tolist(),
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='#60A5FA', width=3),
        marker=dict(size=8, color='#60A5FA'),
        hovertemplate='Cumulative: %{y:.1f}%<extra></extra>'
    ))

    # 80% threshold line
    fig.add_hline(y=80, line_dash="dash", line_color=vital_color, opacity=0.8,
                  annotation_text="80%", yref='y2',
                  annotation_position="right",
                  annotation_font=dict(color=vital_color, size=11))

    # Vital few annotation
    vital_pct = cumulative_pct.iloc[threshold_idx-1] if threshold_idx > 0 else 0

    # Get max y value for proper scaling
    max_y = data.max()

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white'), x=0.5, xanchor='center'),
        xaxis_title='',
        yaxis_title='Count',
        yaxis=dict(
            range=[0, max_y * 1.25],  # Add headroom for labels
            gridcolor='rgba(255,255,255,0.1)',
            title_font=dict(color='white'),
            tickfont=dict(color='white')
        ),
        yaxis2=dict(
            title='Cumulative %',
            overlaying='y',
            side='right',
            range=[0, 105],
            showgrid=False,
            title_font=dict(color='#60A5FA'),
            tickfont=dict(color='#60A5FA')
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(data))),
            ticktext=data.index.tolist(),
            tickangle=-35,
            tickfont=dict(size=10, color='white')
        ),
        height=450,
        showlegend=True,
        legend=dict(
            x=0.5, y=1.12,
            xanchor='center',
            orientation='h',
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=120, l=60, r=60),
        hoverlabel=dict(bgcolor='#1E293B', font_size=12)
    )

    return fig, {'vital_few_count': threshold_idx, 'vital_few_pct': vital_pct}


# ============================================================================
# MAIN APP
# ============================================================================

# Load data
try:
    df = load_data()
    df = categorize_all(df)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run the cleaning notebook first to generate flora-chats-cleaned.csv")
    st.stop()

# Calculate session metrics
session_depths = df.groupby('sessionId').size()
first_messages = df[df['is_first_in_session'] == True] if 'is_first_in_session' in df.columns else df.drop_duplicates('sessionId', keep='first')

# ============================================================================
# PAGE: FLORA PARETO ANALYSIS
# ============================================================================
if PAGES[selected_page] == "pareto":
    # Header
    st.title("Flora Pareto Analysis")
    st.markdown("*Industrial Engineering Analysis of User Interactions with Flora AI*")

    st.markdown("")  # Spacing

    # Key metrics in a clean row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Messages", f"{len(df):,}")
    with col2:
        st.metric("Unique Users", f"{df['userId'].nunique():,}")
    with col3:
        st.metric("Sessions", f"{df['sessionId'].nunique():,}")
    with col4:
        st.metric("Avg Session Depth", f"{session_depths.mean():.1f}")
    with col5:
        high_conf = (df['category_confidence'] == 'high').mean() * 100
        st.metric("Classification Confidence", f"{high_conf:.0f}%")

    st.markdown("")  # Spacing
    st.divider()
    st.markdown("")  # Spacing

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Request Types (Pareto)",
        "👥 User Distribution",
        "🎯 First Prompt Analysis",
        "💬 Engagement Depth",
        "🤖 Model Distribution",
        "🔍 Validation"
    ])

    # TAB 1: REQUEST TYPES PARETO
    with tab1:
        st.header("Pareto Analysis: Request Types")
        st.markdown("*Which 20% of request types drive 80% of Flora usage?*")

        st.markdown("")  # Spacing

        intent_counts = df['intent_category'].value_counts()

        col1, col2 = st.columns([2.5, 1])

        with col1:
            fig, stats = create_pareto_chart(intent_counts, "Flora Request Type Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Pareto Insight")
            st.info(f"**Vital Few:** Top {stats['vital_few_count']} categories = **{stats['vital_few_pct']:.0f}%** of usage")

            st.markdown("")  # Spacing
            st.subheader("Category Breakdown")
            for cat, count in intent_counts.items():
                pct = count / len(df) * 100
                conf_emoji = "🟢" if df[df['intent_category'] == cat]['category_confidence'].mode().iloc[0] == 'high' else "🟡"
                st.write(f"{conf_emoji} **{cat}**: {count} ({pct:.1f}%)")

    # TAB 2: USER DISTRIBUTION PARETO
    with tab2:
        st.header("Pareto Analysis: User Activity")
        st.markdown("*Does the 80/20 rule apply to user engagement?*")

        st.markdown("")  # Spacing

        user_counts = df['userId'].value_counts()
        user_counts_display = user_counts.copy()
        user_counts_display.index = [f'User {i+1}' for i in range(len(user_counts))]

        col1, col2 = st.columns([2.5, 1])

        with col1:
            fig, stats = create_pareto_chart(user_counts_display, "User Activity Distribution", color='#457B9D')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Concentration Insight")
            st.info(f"**Top {stats['vital_few_count']} users** generate **{stats['vital_few_pct']:.0f}%** of all messages")

            st.markdown("")  # Spacing
            st.subheader("User Statistics")
            st.write(f"**Total Users:** {len(user_counts)}")
            st.write(f"**Avg Messages/User:** {user_counts.mean():.1f}")
            st.write(f"**Max Messages:** {user_counts.max()}")
            st.write(f"**Single-message Users:** {(user_counts == 1).sum()}")

    # TAB 3: FIRST PROMPT ANALYSIS
    with tab3:
        st.header("First Prompt Distribution")
        st.markdown("*What do users ask FIRST? This reveals perceived value proposition.*")

        st.markdown("")  # Spacing

        first_prompt_dist = first_messages['intent_category'].value_counts()

        col1, col2 = st.columns([2.5, 1])

        with col1:
            fig, stats = create_pareto_chart(first_prompt_dist, "How Users Start Conversations", color='#2A9D8F')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Adoption Entry Points")
            st.success(f"**Primary Entry:** {first_prompt_dist.index[0]}")

            st.markdown("")  # Spacing
            st.subheader("Top Entry Points")
            for cat, count in first_prompt_dist.head(5).items():
                pct = count / len(first_messages) * 100
                st.write(f"**{cat}**: {count} sessions ({pct:.1f}%)")

    # TAB 4: ENGAGEMENT DEPTH
    with tab4:
        st.header("Engagement Depth Analysis")
        st.markdown("**Are users having conversations or one-shot queries?**")

        st.markdown("")  # Spacing

        col1, col2 = st.columns(2)

        with col1:
            # Session depth histogram
            depth_counts = session_depths.value_counts().sort_index()
            fig = px.bar(x=depth_counts.index, y=depth_counts.values,
                        title="Session Depth Distribution",
                        labels={'x': 'Messages per Session', 'y': 'Number of Sessions'},
                        color_discrete_sequence=['#2E86AB'])
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title=dict(font=dict(color='white', size=16), x=0.5, xanchor='center'),
                xaxis=dict(tickfont=dict(color='white'), title_font=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(tickfont=dict(color='white'), title_font=dict(color='white'), gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Pie chart
            single_msg = (session_depths == 1).sum()
            multi_msg = (session_depths > 1).sum()

            fig = go.Figure(data=[go.Pie(
                labels=['Single Message', 'Multi-turn (2+)'],
                values=[single_msg, multi_msg],
                marker_colors=['#E63946', '#2A9D8F'],
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(color='white', size=12)
            )])
            fig.update_layout(
                title=dict(text="Session Engagement Type", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='white')),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("")  # Spacing

        # Metrics
        st.subheader("Engagement Metrics")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Avg Session Depth", f"{session_depths.mean():.2f} messages")
        with m2:
            st.metric("Single-message Sessions", f"{single_msg/len(session_depths)*100:.0f}%")
        with m3:
            st.metric("Multi-turn Sessions", f"{multi_msg/len(session_depths)*100:.0f}%")

    # TAB 5: MODEL DISTRIBUTION
    with tab5:
        st.header("Model Distribution")
        st.markdown("*AI models powering Flora responses*")

        st.markdown("")  # Spacing

        col1, col2 = st.columns(2)

        with col1:
            # Model provider distribution (pie chart)
            model_counts = df['model'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=model_counts.index.tolist(),
                values=model_counts.values.tolist(),
                marker_colors=['#10B981', '#3B82F6', '#F59E0B'],
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(color='white', size=14)
            )])
            fig.update_layout(
                title=dict(text="Model Provider", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='white')),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Model version detection stats
            st.subheader("Model Details")

            for model, count in model_counts.items():
                pct = count / len(df) * 100
                st.write(f"**{model.upper()}**: {count} messages ({pct:.1f}%)")

            st.markdown("")
            st.divider()
            st.markdown("")

            # Model version info
            if 'model_version' in df.columns:
                versions_detected = df['model_version'].dropna()
                st.subheader("Detected Model Versions")
                if len(versions_detected) > 0:
                    version_counts = versions_detected.value_counts()
                    for version, count in version_counts.items():
                        st.write(f"**{version}**: {count} messages")
                    st.caption(f"*Version detected in {len(versions_detected)}/{len(df)} messages ({len(versions_detected)/len(df)*100:.1f}%)*")
                else:
                    st.info("No specific model versions detected in message content")

        st.markdown("")

        # Additional insights
        st.subheader("Model Usage Insights")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Primary Provider", model_counts.index[0].upper())
        with m2:
            st.metric("Total Providers", len(model_counts))
        with m3:
            if 'model_version' in df.columns:
                versions = df['model_version'].dropna().nunique()
                st.metric("Versions Detected", versions)
            else:
                st.metric("Versions Detected", "N/A")

    # TAB 6: VALIDATION
    with tab6:
        st.header("Category Validation")
        st.markdown("*Review sample classifications to confirm accuracy.*")

        st.markdown("")  # Spacing

        # Confidence breakdown
        conf_counts = df['category_confidence'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Confidence", f"{conf_counts.get('high', 0)} ({conf_counts.get('high', 0)/len(df)*100:.0f}%)")
        with col2:
            st.metric("Medium Confidence", f"{conf_counts.get('medium', 0)} ({conf_counts.get('medium', 0)/len(df)*100:.0f}%)")
        with col3:
            st.metric("Low Confidence", f"{conf_counts.get('low', 0)} ({conf_counts.get('low', 0)/len(df)*100:.0f}%)")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Category selector
        selected_category = st.selectbox("Select category to review:", df['intent_category'].unique())

        samples = df[df['intent_category'] == selected_category].head(5)

        st.subheader(f"Sample: {selected_category}")
        for i, (_, row) in enumerate(samples.iterrows(), 1):
            with st.expander(f"Example {i} - [{row['category_confidence']}]"):
                st.write(f"**Instruction:** {row['instruction'][:300]}..." if len(str(row['instruction'])) > 300 else f"**Instruction:** {row['instruction']}")
                if 'has_json_payload' in row and row['has_json_payload']:
                    st.info("📎 Contains JSON data payload")
                if 'has_reference' in row and row['has_reference']:
                    st.info("🔗 Contains @reference to Flora entity")

# ============================================================================
# PAGE: TOP 4 USER DEEP DIVE
# ============================================================================
else:
    st.title("Top 4 User Deep Dive")
    st.markdown("*Comprehensive analysis of the 4 most active Flora users (94% of all messages)*")

    # Load user analysis data
    user_analysis = load_user_analysis()

    if user_analysis is None:
        st.warning("User analysis data not found. Please run the `quatro_users.ipynb` notebook first.")
        st.info("Run: `uv run jupyter notebook notebooks/quatro_users.ipynb`")
        st.stop()

    st.markdown("")

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top 4 Messages", f"{user_analysis['overview']['top_4_messages']:,}")
    with col2:
        st.metric("% of Total", f"{user_analysis['overview']['top_4_percentage']}%")
    with col3:
        st.metric("Total Messages", f"{user_analysis['overview']['total_messages']:,}")
    with col4:
        st.metric("Analysis Date", user_analysis['overview']['analysis_date'])

    st.markdown("")
    st.divider()
    st.markdown("")

    # Tabs for user analysis
    user_tab1, user_tab2, user_tab3, user_tab4, user_tab5 = st.tabs([
        "📊 Comparison Overview",
        "🔴 User 1: Power User",
        "🔵 User 2: Reporter",
        "🟢 User 3: Sprint Prepper",
        "🟡 User 4: Explorer"
    ])

    # TAB: Comparison Overview
    with user_tab1:
        st.header("User Comparison Overview")

        # Comparison table
        comparison_df = pd.DataFrame(user_analysis['comparison'])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("")

        # Comparison charts
        col1, col2 = st.columns(2)

        with col1:
            # Prompts bar chart
            users = [f"User {i}" for i in range(1, 5)]
            prompts = [user_analysis['user_profiles'][str(i)]['total_prompts'] for i in range(1, 5)]
            colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']

            fig = go.Figure(data=[go.Bar(
                x=users,
                y=prompts,
                marker_color=colors,
                text=prompts,
                textposition='outside'
            )])
            fig.update_layout(
                title=dict(text="Total Prompts by User", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(color='white')),
                yaxis=dict(tickfont=dict(color='white'), title='Count', title_font=dict(color='white')),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Session depth bar chart
            depths = [user_analysis['user_profiles'][str(i)]['avg_session_depth'] for i in range(1, 5)]

            fig = go.Figure(data=[go.Bar(
                x=users,
                y=depths,
                marker_color=colors,
                text=[f"{d:.2f}" for d in depths],
                textposition='outside'
            )])
            fig.update_layout(
                title=dict(text="Avg Session Depth", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(color='white')),
                yaxis=dict(tickfont=dict(color='white'), title='Messages/Session', title_font=dict(color='white')),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        # User 1 vs User 4 contrast
        st.subheader("Key Contrast: User 1 vs User 4")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**User 1: The Power User**")
            u1 = user_analysis['user1_vs_user4']['user1']
            for key, val in u1.items():
                st.write(f"• **{key.title()}**: {val}")

        with col2:
            st.markdown("**User 4: The Explorer**")
            u4 = user_analysis['user1_vs_user4']['user4']
            for key, val in u4.items():
                st.write(f"• **{key.title()}**: {val}")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Combined User 1 + User 4 Pareto Chart
        st.subheader("Combined User 1 + User 4: Intent Distribution Pareto")
        st.markdown("*Analyzing the combined behavior of the most diverse (User 1) and most conversational (User 4) users*")

        combined_pareto_path = DATA_DIR / "user1_user4_combined_pareto.png"
        if combined_pareto_path.exists():
            st.image(str(combined_pareto_path), use_container_width=True)
        else:
            st.warning("Combined Pareto chart not found. Please run the `quatro_users.ipynb` notebook.")

    # Individual user tabs
    def render_user_profile(user_num):
        profile = user_analysis['user_profiles'][str(user_num)]

        # Basic stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Prompts", profile['total_prompts'])
        with col2:
            st.metric("Sessions", profile['unique_sessions'])
        with col3:
            st.metric("Active Days", profile['active_days'])
        with col4:
            st.metric("Avg Depth", f"{profile['avg_session_depth']:.2f}")

        st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Intent Distribution")
            intent_data = profile['intent_distribution']

            # Pareto chart for this user
            intent_series = pd.Series(intent_data).sort_values(ascending=False)
            fig, stats = create_pareto_chart(intent_series, f"User {user_num} Intent Pareto")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Profile Details")
            st.write(f"**Persona:** {profile['user_name']}")
            st.write(f"**Date Range:** {profile['first_activity']} to {profile['last_activity']}")
            st.write(f"**Top Intent:** {profile['top_intent']} ({profile['top_intent_pct']}%)")
            st.write(f"**Intent Diversity:** {profile['intent_categories_used']} categories")
            st.write(f"**Single-msg Sessions:** {profile['single_msg_sessions_pct']}%")
            st.write(f"**High Confidence:** {profile['high_confidence_pct']}%")
            st.write(f"**Peak Day:** {profile['peak_day']} ({profile['peak_day_prompts']} prompts)")

        st.markdown("")
        st.subheader("All Prompts")

        prompts_df = pd.DataFrame(profile['all_prompts'])
        st.dataframe(prompts_df, use_container_width=True, hide_index=True)

    with user_tab2:
        st.header("User 1: The Power User")
        st.markdown("*Experienced user with diverse, sophisticated queries across multiple teams*")
        render_user_profile(1)

    with user_tab3:
        st.header("User 2: The Reporter")
        st.markdown("*Heavy executive reporting usage with templated queries*")
        render_user_profile(2)

    with user_tab4:
        st.header("User 3: The Sprint Prepper")
        st.markdown("*Focused on sprint preparation and retrospective analysis*")
        render_user_profile(3)

    with user_tab5:
        st.header("User 4: The Explorer")
        st.markdown("*New user learning through conversational iteration*")
        render_user_profile(4)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("")
st.divider()
st.caption(
    "**BLOOM-02** | Designing AI Agentic Structures for SDLC Project Management | "
    "Bloomfilter AI | Industrial Engineering Pareto Analysis"
)
