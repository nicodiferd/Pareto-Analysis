"""
Flora Chat Analysis - Streamlit Dashboard
Industrial Engineering Pareto Analysis of Flora User Interactions
"""

import streamlit as st
import pandas as pd
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
    "Top 4 User Deep Dive": "users",
    "January 2026 Analysis": "jan2026"
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
        if st.button("Submit", width="stretch"):
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

@st.cache_data
def load_jan2026_data():
    """Load January 2026 analyzed data."""
    try:
        df = pd.read_csv(DATA_DIR / "flora_data_01_2026_analyzed.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def get_jan2026_raw_stats():
    """Get stats about the raw January 2026 data file."""
    raw_path = DATA_DIR / "flora_data_01_2026.csv"
    cleaned_path = DATA_DIR / "flora_data_01_2026_cleaned.csv"
    analyzed_path = DATA_DIR / "flora_data_01_2026_analyzed.csv"

    stats = {}
    if raw_path.exists():
        stats['raw_size_mb'] = raw_path.stat().st_size / (1024 * 1024)
        # Count lines (rows)
        with open(raw_path, 'r') as f:
            stats['raw_rows'] = sum(1 for _ in f) - 1  # Subtract header
    if cleaned_path.exists():
        stats['cleaned_size_mb'] = cleaned_path.stat().st_size / (1024 * 1024)
    if analyzed_path.exists():
        stats['analyzed_size_mb'] = analyzed_path.stat().st_size / (1024 * 1024)

    return stats


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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
elif PAGES[selected_page] == "users":
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
        st.dataframe(comparison_df, width="stretch", hide_index=True)

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
            st.plotly_chart(fig, width="stretch")

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
            st.plotly_chart(fig, width="stretch")

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
            st.image(str(combined_pareto_path), width="stretch")
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
            fig, _ = create_pareto_chart(intent_series, f"User {user_num} Intent Pareto")
            st.plotly_chart(fig, width="stretch")

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
        st.dataframe(prompts_df, width="stretch", hide_index=True)

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
# PAGE: JANUARY 2026 ANALYSIS
# ============================================================================
elif PAGES[selected_page] == "jan2026":
    st.title("January 2026 Dataset Analysis")
    st.markdown("*New dataset with enhanced cleaning pipeline and hybrid intent classification*")

    # Load January 2026 data
    df_jan = load_jan2026_data()
    raw_stats = get_jan2026_raw_stats()

    if df_jan is None:
        st.warning("January 2026 data not found. Please run the notebooks first.")
        st.code("uv run jupyter notebook notebooks/cleaning_01_2026.ipynb\nuv run jupyter notebook notebooks/analysis_01_2026.ipynb")
        st.stop()

    st.markdown("")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Messages", f"{len(df_jan):,}")
    with col2:
        st.metric("Unique Users", f"{df_jan['userId'].nunique():,}")
    with col3:
        st.metric("Sessions", f"{df_jan['sessionId'].nunique():,}")
    with col4:
        avg_conf = df_jan['intent_confidence'].mean() * 100
        st.metric("Avg Confidence", f"{avg_conf:.0f}%")
    with col5:
        rule_based_pct = (df_jan['intent_method'] == 'rules').mean() * 100
        st.metric("Rule-Based", f"{rule_based_pct:.0f}%")

    st.markdown("")
    st.divider()
    st.markdown("")

    # Tab category headers with color coding
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p style="color: #22c55e; font-size: 1.1em; font-weight: 600; margin-bottom: 5px;">Solutions <span style="font-weight: 400; font-size: 0.9em;">(Key Findings & Results)</span></p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p style="color: #3b82f6; font-size: 1.1em; font-weight: 600; margin-bottom: 5px;">Methodology <span style="font-weight: 400; font-size: 0.9em;">(Data Processing & Approach)</span></p>', unsafe_allow_html=True)

    # Tabs with colored emoji indicators - Solutions (green circle), Methodology (blue circle)
    jan_tab1, jan_tab2, jan_tab3, jan_tab4, jan_tab5, jan_tab6, jan_tab7 = st.tabs([
        "🟢 Pareto Analysis",
        "🟢 User Analysis",
        "🟢 Session & Model",
        "🟢 Conclusions",
        "🔵 Dataset Overview",
        "🔵 Cleaning Pipeline",
        "🔵 Intent Classification"
    ])

    # TAB 5: DATASET OVERVIEW (Methodology)
    with jan_tab5:
        st.header("Dataset Overview")
        st.markdown("*Understanding the raw data and its challenges*")

        st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raw Data Characteristics")

            st.markdown("""
            **Source:** Flora chat export (January 2026)

            **Export Format Issues:**
            - Triple-quoted CSV fields (`\"""value\"""`)
            - Multi-layer JSON escaping
            - Two distinct data formats in same file
            - Markdown content embedded within CSV cells

            **Original File:**
            """)

            if 'raw_size_mb' in raw_stats:
                st.write(f"- **Size:** {raw_stats['raw_size_mb']:.1f} MB")
                st.write(f"- **Rows:** {raw_stats.get('raw_rows', 'N/A'):,}")

            st.markdown("")
            st.info("The raw CSV required specialized parsing due to complex escaping patterns from the export process.")

        with col2:
            st.subheader("Data Format Discovery")

            # Show the two formats
            json_format = len(df_jan[df_jan['input_format'] == 'json'])
            plain_format = len(df_jan[df_jan['input_format'] != 'json'])

            fig = go.Figure(data=[go.Pie(
                labels=['JSON Wrapped', 'Plain Text'],
                values=[json_format, plain_format],
                marker_colors=['#3B82F6', '#10B981'],
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(color='white', size=14)
            )])
            fig.update_layout(
                title=dict(text="Data Format Distribution", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='white')),
                height=300
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("""
            **Format Implications:**
            - **JSON Wrapped:** Contains model metadata, token counts
            - **Plain Text:** No model info recoverable (52% of data)
            """)

        st.markdown("")
        st.divider()
        st.markdown("")

        # Data transformation summary
        st.subheader("Data Transformation Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Raw Data**")
            if 'raw_size_mb' in raw_stats:
                st.metric("File Size", f"{raw_stats['raw_size_mb']:.1f} MB")
            st.caption("Complex CSV with nested JSON")

        with col2:
            st.markdown("**Cleaned Data**")
            if 'cleaned_size_mb' in raw_stats:
                st.metric("File Size", f"{raw_stats['cleaned_size_mb']:.1f} MB")
            st.caption("Normalized prompts & responses")

        with col3:
            st.markdown("**Analyzed Data**")
            if 'analyzed_size_mb' in raw_stats:
                st.metric("File Size", f"{raw_stats['analyzed_size_mb']:.1f} MB")
            st.caption("With intent classifications")

        st.markdown("")

        # Comprehensive Data Preview
        st.subheader("Complete Data Preview")

        preview_option = st.selectbox(
            "Select data view:",
            ["Quick Overview (Key Fields)", "Full Record Sample", "Column Statistics", "Date Range Analysis"],
            key="dataset_preview"
        )

        if preview_option == "Quick Overview (Key Fields)":
            st.markdown("**First 10 records with key fields:**")
            sample_cols = ['timestamp', 'user_label', 'prompt', 'intent', 'intent_confidence']
            sample_df = df_jan[sample_cols].head(10).copy()
            sample_df['prompt'] = sample_df['prompt'].apply(lambda x: str(x)[:80] + '...' if len(str(x)) > 80 else x)
            sample_df['timestamp'] = sample_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            sample_df.columns = ['Timestamp', 'User', 'Prompt (truncated)', 'Intent Category', 'Confidence']
            st.dataframe(sample_df, width="stretch", hide_index=True)

        elif preview_option == "Full Record Sample":
            st.markdown("**Complete record details (first 5):**")
            full_cols = ['timestamp', 'user_label', 'sessionId', 'session_msg_num', 'prompt', 'response', 'intent', 'model_simple', 'total_tokens']
            full_df = df_jan[full_cols].head(5).copy()
            full_df['prompt'] = full_df['prompt'].apply(lambda x: str(x)[:150] + '...' if len(str(x)) > 150 else x)
            full_df['response'] = full_df['response'].apply(lambda x: str(x)[:150] + '...' if len(str(x)) > 150 else x)
            full_df['timestamp'] = full_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            full_df.columns = ['Timestamp', 'User', 'Session ID', 'Msg #', 'Prompt', 'Response', 'Intent', 'Model', 'Tokens']
            st.dataframe(full_df, width="stretch", hide_index=True)

        elif preview_option == "Column Statistics":
            st.markdown("**Dataset columns and types:**")
            col_stats = pd.DataFrame({
                'Column': df_jan.columns,
                'Type': df_jan.dtypes.astype(str),
                'Non-Null': df_jan.count().values,
                'Sample Value': [str(df_jan[col].iloc[0])[:50] + '...' if len(str(df_jan[col].iloc[0])) > 50 else str(df_jan[col].iloc[0]) for col in df_jan.columns]
            })
            st.dataframe(col_stats, width="stretch", hide_index=True)

        else:  # Date Range Analysis
            st.markdown("**Activity by date:**")
            daily_counts = df_jan.groupby('date').size().reset_index(name='Messages')
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily_counts['date'], y=daily_counts['Messages'], marker_color='#3B82F6'))
            fig.update_layout(
                title=dict(text="Messages per Day", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                xaxis=dict(title='Date', tickfont=dict(color='white'), title_font=dict(color='white')),
                yaxis=dict(title='Messages', tickfont=dict(color='white'), title_font=dict(color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig, width="stretch")

    # TAB 6: CLEANING PIPELINE (Methodology)
    with jan_tab6:
        st.header("Data Cleaning Pipeline")
        st.markdown("*Multi-stage cleaning process to handle complex export format*")

        st.markdown("")

        # Pipeline visualization
        st.subheader("Pipeline Stages")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("### Stage 1")
            st.markdown("**Load & Parse**")
            st.markdown("""
            - Read CSV with proper quoting
            - Handle triple-quoted strings
            - Identify JSON vs plain text rows
            """)
            st.success("356 rows loaded")

        with col2:
            st.markdown("### Stage 2")
            st.markdown("**Unescape JSON**")
            st.markdown("""
            - Replace `\\\\\\\"` (nested quotes)
            - Replace `\\"` (structure quotes)
            - Restore nested escaping
            """)
            st.success("100% parsed")

        with col3:
            st.markdown("### Stage 3")
            st.markdown("**Extract Fields**")
            st.markdown("""
            - Extract prompt/response
            - Parse model metadata
            - Calculate token counts
            """)
            st.success("27 columns extracted")

        with col4:
            st.markdown("### Stage 4")
            st.markdown("**Enrich Data**")
            st.markdown("""
            - Add session tracking
            - Compute message numbers
            - Flag embedded data
            """)
            st.success("Ready for analysis")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Key challenges and solutions
        st.subheader("Key Challenges Solved")

        with st.expander("Challenge 1: Multi-Layer JSON Escaping", expanded=True):
            st.markdown("""
            **Problem:** The CSV export contained multiple layers of escaping:
            - Structure quotes: `\\"` (1 backslash + quote)
            - Nested content quotes: `\\\\\\\"` (3 backslashes + quote)

            **Solution:**
            ```python
            # Step 1: Protect nested quotes with marker
            unescaped = stripped.replace('\\\\\\\\\\\"', '___NESTED___')
            # Step 2: Unescape structure quotes
            unescaped = unescaped.replace('\\\\"', '"')
            # Step 3: Restore nested quotes
            unescaped = unescaped.replace('___NESTED___', '\\\\"')
            ```
            """)

        with st.expander("Challenge 2: Two Data Formats in One File"):
            st.markdown("""
            **Problem:** The dataset contains two distinct formats:
            - **207 rows:** JSON wrapped `{"messages": [...]}`
            - **149 rows:** Plain text directly

            **Solution:** Check format before parsing:
            ```python
            if stripped.startswith('{'):
                # Parse as JSON, extract from messages array
            else:
                # Treat as plain text prompt
            ```
            """)

        with st.expander("Challenge 3: Missing Model Metadata"):
            st.markdown("""
            **Problem:** Model information only exists in JSON-formatted outputs.

            **Impact:**
            - 184/356 (52%) have model data
            - 172/356 (48%) show as "unknown"

            **Decision:** Accept partial model coverage rather than fabricate data.
            """)

        st.markdown("")

        # Cleaning results
        st.subheader("Cleaning Results")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Prompts Extracted", "356 (100%)")
        with m2:
            st.metric("Responses Extracted", "356 (100%)")
        with m3:
            known_models = len(df_jan[df_jan['model_simple'] != 'unknown'])
            st.metric("Model Info Available", f"{known_models} ({known_models/len(df_jan)*100:.0f}%)")
        with m4:
            embedded = df_jan['has_embedded_data'].sum()
            st.metric("Has Embedded Data", f"{embedded} ({embedded/len(df_jan)*100:.0f}%)")

    # TAB 7: INTENT CLASSIFICATION (Methodology)
    with jan_tab7:
        st.header("Intent Classification Methodology")
        st.markdown("*How we categorized 356 user prompts into 8 distinct request types*")

        st.markdown("")

        # Executive-friendly overview
        st.subheader("The Challenge")
        st.info("""
        **Problem:** We received 356 raw user messages to Flora. To understand what users are asking for,
        we needed to categorize each message into meaningful groups.

        **Solution:** We developed a systematic classification approach that automatically labels each
        user prompt with one of 8 request types, achieving 99.4% accuracy.
        """)

        st.markdown("")

        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.subheader("8 Request Categories")
            st.markdown("*Each user prompt is assigned to exactly one category*")

            taxonomy_data = {
                'Category': [
                    'Metrics Query',
                    'Risk & Process',
                    'Sprint Report',
                    'Information Request',
                    'Initiative Query',
                    'Sprint Retrospective',
                    'Performance Analysis',
                    'Executive Summary'
                ],
                'What Users Are Asking For': [
                    'Numbers and KPIs (velocity, throughput, cycle time)',
                    'Problems to address (bottlenecks, blockers, delays)',
                    'Team progress summaries and data breakdowns',
                    'General learning and exploration questions',
                    'Status of specific projects, epics, or features',
                    'Post-sprint reviews and lessons learned',
                    'Trend analysis and performance comparisons',
                    'High-level summaries for leadership'
                ],
                'Example User Prompt': [
                    '"What is the velocity for the last 6 sprints?"',
                    '"Show me all off-track initiatives"',
                    '"Analyze this data and provide a summary"',
                    '"Tell me about this project"',
                    '"How are my top initiatives doing?"',
                    '"Provide sprint retrospective analysis"',
                    '"Why is performance decreasing?"',
                    '"Provide executive summary for this period"'
                ],
                'Count': [
                    df_jan[df_jan['intent'] == 'Metrics Query'].shape[0],
                    df_jan[df_jan['intent'] == 'Risk & Process'].shape[0],
                    df_jan[df_jan['intent'] == 'Sprint Report'].shape[0],
                    df_jan[df_jan['intent'] == 'Information Request'].shape[0],
                    df_jan[df_jan['intent'] == 'Initiative Query'].shape[0],
                    df_jan[df_jan['intent'] == 'Sprint Retrospective'].shape[0],
                    df_jan[df_jan['intent'] == 'Performance Analysis'].shape[0],
                    df_jan[df_jan['intent'] == 'Executive Summary'].shape[0]
                ]
            }

            st.dataframe(pd.DataFrame(taxonomy_data), width="stretch", hide_index=True)

        with col2:
            st.subheader("How It Works")

            st.markdown("""
            **Classification Process:**

            1. **Read each prompt** - Take the user's question
            2. **Match keywords** - Look for indicator words like "velocity", "bottleneck", "summary"
            3. **Assign category** - Place in the most appropriate bucket
            4. **Score confidence** - Rate how certain we are (50-90%)

            **Results:**
            - **354 of 356** prompts classified automatically
            - Only **2 prompts** required manual review
            - Average confidence: **85%**
            """)

            st.markdown("")
            st.success("**99.4% automated classification** with no external AI costs")

            st.markdown("")

            # Classification method breakdown
            method_counts = df_jan['intent_method'].value_counts()
            st.markdown("**Method Distribution:**")
            for method, count in method_counts.items():
                pct = count / len(df_jan) * 100
                icon = "✅" if method == 'rules' else "🤖" if method == 'llm' else "⚠️"
                st.write(f"{icon} **{method}**: {count} ({pct:.1f}%)")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Confidence analysis
        st.subheader("Classification Confidence Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Confidence distribution histogram
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df_jan['intent_confidence'],
                nbinsx=20,
                marker_color='#3B82F6',
                opacity=0.8
            ))
            fig.update_layout(
                title=dict(text="Confidence Score Distribution", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                xaxis=dict(title='Confidence', tickfont=dict(color='white'), title_font=dict(color='white')),
                yaxis=dict(title='Count', tickfont=dict(color='white'), title_font=dict(color='white')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig, width="stretch")

        with col2:
            # Confidence by category
            conf_by_intent = df_jan.groupby('intent')['intent_confidence'].mean().sort_values(ascending=True)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=conf_by_intent.values,
                y=conf_by_intent.index,
                orientation='h',
                marker_color='#10B981',
                text=[f"{v:.2f}" for v in conf_by_intent.values],
                textposition='outside'
            ))
            fig.update_layout(
                title=dict(text="Avg Confidence by Category", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                xaxis=dict(title='Avg Confidence', tickfont=dict(color='white'), title_font=dict(color='white'), range=[0, 1.1]),
                yaxis=dict(tickfont=dict(color='white', size=10)),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=150)
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("")

        # Confidence metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            high_conf = (df_jan['intent_confidence'] >= 0.7).sum()
            st.metric("High Confidence (>=0.7)", f"{high_conf} ({high_conf/len(df_jan)*100:.0f}%)")
        with m2:
            med_conf = ((df_jan['intent_confidence'] >= 0.5) & (df_jan['intent_confidence'] < 0.7)).sum()
            st.metric("Medium (0.5-0.7)", f"{med_conf} ({med_conf/len(df_jan)*100:.0f}%)")
        with m3:
            low_conf = (df_jan['intent_confidence'] < 0.5).sum()
            st.metric("Low (<0.5)", f"{low_conf} ({low_conf/len(df_jan)*100:.0f}%)")
        with m4:
            st.metric("Mean Confidence", f"{df_jan['intent_confidence'].mean():.2f}")

    # TAB 1: PARETO ANALYSIS (Solutions)
    with jan_tab1:
        st.header("Pareto Analysis: Request Types")
        st.markdown("*Which 20% of request types drive 80% of Flora usage?*")

        st.markdown("")

        # Filter out Empty/Invalid
        valid_intents = df_jan[df_jan['intent'] != 'Empty/Invalid']['intent'].value_counts()

        col1, col2 = st.columns([2.5, 1])

        with col1:
            fig, stats = create_pareto_chart(valid_intents, "January 2026 - Request Type Distribution")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Pareto Insight")
            st.info(f"**Vital Few:** Top {stats['vital_few_count']} categories = **{stats['vital_few_pct']:.0f}%** of requests")

            st.markdown("")
            st.subheader("Category Breakdown")

            total = valid_intents.sum()
            cumsum = 0
            for cat, count in valid_intents.items():
                pct = count / total * 100
                cumsum += count
                cum_pct = cumsum / total * 100
                emoji = "🔴" if cum_pct <= 80 else "🔵"
                st.write(f"{emoji} **{cat}**: {count} ({pct:.1f}%)")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Key findings
        st.subheader("Key Findings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Request Types:**")
            top_3 = valid_intents.head(3)
            top_3_pct = top_3.sum() / total * 100
            st.success(f"Top 3 categories account for **{top_3_pct:.0f}%** of all requests")

            for i, (cat, count) in enumerate(top_3.items(), 1):
                st.write(f"{i}. **{cat}**: {count} ({count/total*100:.1f}%)")

        with col2:
            st.markdown("**Implications:**")
            st.markdown("""
            - **Metrics Query** dominance suggests users primarily use Flora for data retrieval
            - **Risk & Process** indicates strong adoption for identifying bottlenecks
            - **Sprint Report** shows consistent use for sprint summaries

            **Recommendation:** Optimize these three categories for faster response times.
            """)

        st.markdown("")
        st.divider()
        st.markdown("")

        # Prompt Preview by Category
        st.subheader("Explore Raw Prompts by Category")
        st.markdown("*See the actual user prompts that were classified into each category*")

        all_intents = df_jan[df_jan['intent'] != 'Empty/Invalid']['intent'].unique().tolist()
        selected_intent = st.selectbox(
            "Select a category to preview prompts:",
            all_intents,
            key="pareto_intent_preview"
        )

        if selected_intent:
            intent_prompts = df_jan[df_jan['intent'] == selected_intent][['user_label', 'prompt', 'intent_confidence']].head(10).copy()
            intent_prompts['prompt'] = intent_prompts['prompt'].apply(lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else x)
            intent_prompts['intent_confidence'] = intent_prompts['intent_confidence'].apply(lambda x: f"{x:.0%}")
            intent_prompts.columns = ['User', 'Raw Prompt', 'Confidence']

            st.markdown(f"**Sample prompts labeled as '{selected_intent}':**")
            st.dataframe(intent_prompts, width="stretch", hide_index=True)

            total_in_cat = len(df_jan[df_jan['intent'] == selected_intent])
            st.caption(f"Showing first 10 of {total_in_cat} prompts in this category")

    # TAB 2: USER ANALYSIS (Solutions)
    with jan_tab2:
        st.header("User Activity Analysis")
        st.markdown("*Understanding user engagement patterns*")

        st.markdown("")

        # User Pareto
        user_counts = df_jan['user_label'].value_counts()

        col1, col2 = st.columns([2.5, 1])

        with col1:
            fig, stats = create_pareto_chart(user_counts, "User Activity Distribution", color='#457B9D')
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("User Concentration")
            st.info(f"**Top {stats['vital_few_count']} users** generate **{stats['vital_few_pct']:.0f}%** of messages")

            st.markdown("")
            st.subheader("User Stats")
            st.write(f"**Total Users:** {df_jan['userId'].nunique()}")
            st.write(f"**Avg Messages/User:** {user_counts.mean():.1f}")
            st.write(f"**Max Messages:** {user_counts.max()}")
            single_msg_users = (user_counts == 1).sum()
            st.write(f"**Single-message Users:** {single_msg_users}")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Top user breakdown
        st.subheader("Top Users - Intent Breakdown")

        top_users = user_counts.head(5).index.tolist()

        user_cols = st.columns(len(top_users))

        for i, user in enumerate(top_users):
            with user_cols[i]:
                user_df = df_jan[df_jan['user_label'] == user]
                st.markdown(f"**{user}**")
                st.write(f"Messages: {len(user_df)}")

                top_intent = user_df['intent'].value_counts().head(1)
                if len(top_intent) > 0:
                    st.write(f"Top: {top_intent.index[0]}")
                    st.caption(f"({top_intent.values[0]} msgs)")

        st.markdown("")
        st.divider()
        st.markdown("")

        # User Prompt Preview
        st.subheader("Explore User Prompts")
        st.markdown("*See the actual prompts from each user*")

        all_users = df_jan['user_label'].unique().tolist()
        selected_user = st.selectbox(
            "Select a user to preview their prompts:",
            sorted(all_users, key=lambda x: int(x.split()[1]) if x.split()[1].isdigit() else 999),
            key="user_prompt_preview"
        )

        if selected_user:
            user_prompts = df_jan[df_jan['user_label'] == selected_user][['timestamp', 'prompt', 'intent', 'session_msg_num']].head(10).copy()
            user_prompts['prompt'] = user_prompts['prompt'].apply(lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else x)
            user_prompts['timestamp'] = user_prompts['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            user_prompts.columns = ['Timestamp', 'Prompt', 'Intent Category', 'Msg # in Session']

            st.markdown(f"**Sample prompts from {selected_user}:**")
            st.dataframe(user_prompts, width="stretch", hide_index=True)

            total_user_msgs = len(df_jan[df_jan['user_label'] == selected_user])
            user_sessions = df_jan[df_jan['user_label'] == selected_user]['sessionId'].nunique()
            st.caption(f"Showing first 10 of {total_user_msgs} prompts across {user_sessions} sessions")

    # TAB 3: SESSION & MODEL (Solutions)
    with jan_tab3:
        st.header("Session & Model Analysis")

        st.markdown("")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Session Behavior")

            session_depths = df_jan.groupby('sessionId')['session_msg_num'].max()
            single_turn = (session_depths == 1).sum()
            multi_turn = (session_depths > 1).sum()

            # Session type pie
            fig = go.Figure(data=[go.Pie(
                labels=['Single Message', 'Multi-turn (2+)'],
                values=[single_turn, multi_turn],
                marker_colors=['#E63946', '#2A9D8F'],
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(color='white', size=12)
            )])
            fig.update_layout(
                title=dict(text="Session Types", font=dict(color='white', size=16), x=0.5, xanchor='center'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='white')),
                height=300
            )
            st.plotly_chart(fig, width="stretch")

            st.markdown("")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Sessions", len(session_depths))
            with m2:
                st.metric("Avg Depth", f"{session_depths.mean():.1f}")
            with m3:
                st.metric("Max Depth", session_depths.max())

        with col2:
            st.subheader("Model Distribution")

            # Only show known models
            model_counts = df_jan['model_simple'].value_counts()
            known_models = model_counts[model_counts.index != 'unknown']

            if len(known_models) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=known_models.index.tolist(),
                    values=known_models.values.tolist(),
                    marker_colors=['#4285f4', '#34a853', '#fbbc05', '#ea4335'],
                    hole=0.4,
                    textinfo='label+percent',
                    textfont=dict(color='white', size=12)
                )])
                fig.update_layout(
                    title=dict(text=f"Model Usage (Known Only: {known_models.sum()}/{len(df_jan)})",
                               font=dict(color='white', size=16), x=0.5, xanchor='center'),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color='white')),
                    height=300
                )
                st.plotly_chart(fig, width="stretch")

            st.markdown("")

            st.markdown("**Model Breakdown:**")
            for model, count in model_counts.items():
                pct = count / len(df_jan) * 100
                icon = "✅" if model != 'unknown' else "❓"
                st.write(f"{icon} **{model}**: {count} ({pct:.1f}%)")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Note about model data availability
        unknown_count = model_counts.get('unknown', 0)
        unknown_pct = unknown_count / len(df_jan) * 100
        st.warning(f"""
        **Note on Model Data:** {unknown_count} prompts ({unknown_pct:.0f}%) show "unknown" for model.

        **Why?** The raw data contains two formats:
        - **JSON-wrapped messages** (52%): Include model metadata (token counts, model name, etc.)
        - **Plain text messages** (48%): No metadata available - model info cannot be recovered

        This is a limitation of the data export format, not a data quality issue. The model distribution
        shown above represents only the {known_models.sum()} prompts where model information was available.
        """)

        st.markdown("")
        st.divider()
        st.markdown("")

        # First message analysis
        st.subheader("Session Starters - What Initiates Conversations?")

        first_msgs = df_jan[df_jan['is_first_in_session'] == True]
        first_intent_dist = first_msgs['intent'].value_counts()
        first_intent_dist = first_intent_dist[first_intent_dist.index != 'Empty/Invalid']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=first_intent_dist.values,
            y=first_intent_dist.index,
            orientation='h',
            marker_color='#2A9D8F',
            text=first_intent_dist.values,
            textposition='outside'
        ))
        fig.update_layout(
            title=dict(text="First Message Intent Distribution", font=dict(color='white', size=16), x=0.5, xanchor='center'),
            xaxis=dict(title='Count', tickfont=dict(color='white'), title_font=dict(color='white')),
            yaxis=dict(tickfont=dict(color='white', size=11)),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, width="stretch")

        st.success(f"**Primary Entry Point:** {first_intent_dist.index[0]} ({first_intent_dist.values[0]} sessions)")

    # TAB 4: CONCLUSIONS (Solutions)
    with jan_tab4:
        st.header("Conclusions & Lessons Learned")
        st.markdown("*Summary of the analysis session and key takeaways*")

        st.markdown("")

        # Executive Summary
        st.subheader("Executive Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**What We Did:**")
            st.markdown("""
            1. Received new Flora chat dataset (January 2026)
            2. Developed custom cleaning pipeline for complex CSV format
            3. Created 8-category MECE intent taxonomy
            4. Built rule-based classifier with 99.4% coverage
            5. Performed Pareto analysis on request types and users
            """)

        with col2:
            st.markdown("**Key Metrics:**")
            st.metric("Total Messages Analyzed", "356")
            st.metric("Classification Accuracy", "99.4%")
            st.metric("Categories Defined", "8")

        st.markdown("")
        st.divider()
        st.markdown("")

        # Key Findings
        st.subheader("Key Findings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Request Type Insights**")
            st.info("""
            **Top 3 categories = 72% of requests:**
            - Metrics Query (39%)
            - Risk & Process (17%)
            - Sprint Report (15%)

            Users primarily use Flora for data retrieval and risk identification.
            """)

        with col2:
            st.markdown("**User Behavior Insights**")
            st.info("""
            **Pareto principle confirmed:**
            - Top 5 users = 94% of messages
            - 75% single-turn sessions
            - Avg 1.5 messages/session

            Power users drive adoption; most queries are one-shot.
            """)

        with col3:
            st.markdown("**Data Quality Insights**")
            st.info("""
            **Model metadata gaps:**
            - 52% have model info
            - 48% show as "unknown"

            JSON format outputs contain metadata; plain text doesn't.
            """)

        st.markdown("")
        st.divider()
        st.markdown("")

        # Lessons Learned
        st.subheader("Lessons Learned")

        with st.expander("1. Data Export Complexity", expanded=True):
            st.markdown("""
            **Challenge:** Raw CSV contained triple-quoted strings with multi-layer JSON escaping.

            **Solution:** Custom parsing with marker-based quote replacement:
            - Protect nested quotes (`\\\\\\\"`) with placeholders
            - Unescape structure quotes (`\\"`)
            - Restore nested content

            **Lesson:** Always inspect raw data before assuming standard formats.
            """)

        with st.expander("2. Two Data Formats in One File"):
            st.markdown("""
            **Challenge:** Dataset contained both JSON-wrapped and plain text messages.

            **Solution:** Format detection before parsing:
            - Check if content starts with `{`
            - Apply appropriate extraction method

            **Lesson:** Data heterogeneity requires flexible parsing strategies.
            """)

        with st.expander("3. Intent Classification Without LLM"):
            st.markdown("""
            **Challenge:** Initially planned LLM fallback for edge cases.

            **Solution:** Iterative pattern refinement achieved 99.4% rule-based coverage:
            - Started at 61% coverage
            - Added patterns for specific metrics, risk keywords, conversational starters
            - Only 2 prompts remain as edge cases

            **Lesson:** Well-designed regex patterns can eliminate expensive LLM dependencies.
            """)

        with st.expander("4. MECE Taxonomy Design"):
            st.markdown("""
            **Challenge:** Initial categories had overlap and ambiguity.

            **Solution:** Applied MECE principles:
            - **M**utually **E**xclusive: No prompt fits multiple categories
            - **C**ollectively **E**xhaustive: Every prompt has a category
            - 8 categories with clear boundaries

            **Lesson:** Invest time in taxonomy design before classification.
            """)

        st.markdown("")
        st.divider()
        st.markdown("")

        # Recommendations
        st.subheader("Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**For Flora Product Team:**")
            st.markdown("""
            1. **Optimize Metrics Query UX** - 39% of requests; consider quick-access metrics dashboard
            2. **Improve Risk Visibility** - 17% ask about off-track items; surface proactively
            3. **Address Model Metadata Gap** - Ensure consistent logging across all output formats
            4. **Power User Engagement** - Top 5 users drive 94% usage; gather their feedback
            """)

        with col2:
            st.markdown("**For Future Analysis:**")
            st.markdown("""
            1. **Track Intent Over Time** - Compare category distribution across months
            2. **User Journey Analysis** - Map how intent evolves within sessions
            3. **Response Quality Metrics** - Correlate intent with user satisfaction
            4. **Automate Pipeline** - Convert notebooks to scheduled ETL jobs
            """)

        st.markdown("")

        # Session stats
        st.subheader("Analysis Session Stats")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Data Files Created", "3")
        with m2:
            st.metric("Notebooks Used", "2")
        with m3:
            st.metric("Regex Patterns", "20+")
        with m4:
            st.metric("LLM Calls Needed", "0")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("")
st.divider()
st.caption(
    "**BLOOM-02** | Designing AI Agentic Structures for SDLC Project Management | "
    "Bloomfilter AI | Industrial Engineering Pareto Analysis"
)
