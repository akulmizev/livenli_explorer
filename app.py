import streamlit as st
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="LiveNLI Explanation Explorer",
    page_icon="🔬",
    layout="centered",
)

# --- Custom CSS for Styling ---
# We'll inject some CSS to make our prediction cards look nicer.
# st.markdown("""
# <style>
#     /* Style for the prediction cards */
#     .prediction-card {
#         border: 1px solid #e1e1e1;
#         border-radius: 10px;
#         padding: 20px;
#         margin-bottom: 20px;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         transition: box-shadow 0.3s ease-in-out;
#     }
#     .prediction-card:hover {
#         box-shadow: 0 8px 16px rgba(0,0,0,0.2);
#     }
#     /* Style for the label text */
#     .label-true { color: #28a745; font-weight: bold; }
#     .label-false { color: #dc3545; font-weight: bold; }
#     .label-either { color: #007bff; font-weight: bold; }
# </style>
# """, unsafe_allow_html=True)


# --- Load Data (with caching) ---
@st.cache_data
def load_data():
    try:
        pairs_df = pd.read_csv('sentence_pairs.csv')
        preds_df = pd.read_csv('predictions.csv')
        # Pre-calculate counts for the dashboard
        human_count = preds_df[preds_df['participant_type'] == 'human'].shape[0]
        model_count = preds_df[preds_df['participant_type'] != 'human'].shape[0]
        return pairs_df, preds_df, human_count, model_count
    except FileNotFoundError:
        st.error("Make sure `sentence_pairs.csv` and `predictions.csv` are in the same directory.")
        return None, None, 0, 0


pairs_df, preds_df, human_count, model_count = load_data()

# Stop the app if data isn't loaded
if pairs_df is None:
    st.stop()

# --- Sidebar for Navigation & Filters ---
with st.sidebar:
    st.title(" NLI Explorer")
    st.markdown("---")

    st.header("1. Select a Sentence Pair")

    # Allow user to choose between searching and browsing
    search_method = st.radio(
        "How do you want to find a pair?",
        ('Browse from a list', 'Search by ID'),
        label_visibility="collapsed"
    )

    all_ids = pairs_df['sent_id'].tolist()
    selected_id = None

    if search_method == 'Browse from a list':
        selected_id = st.selectbox(
            "Choose a `sent_id` to inspect:",
            options=sorted(all_ids),
            index=0  # Default to the first item
        )
    else:
        search_id = st.text_input("Enter a `sent_id` (e.g., 20274n):", "")
        if search_id:
            if search_id in all_ids:
                selected_id = search_id
            else:
                st.warning("ID not found. Please try another.")

    st.markdown("---")
    st.header("2. Filter Predictions")

    # Filter by Participant Type
    participant_types = preds_df['participant_type'].unique()
    selected_types = st.multiselect(
        "Toggle predictions by participant type:",
        options=participant_types,
        default=participant_types
    )
    st.markdown("---")

# --- Main Page ---
st.title("📊 NLI Dashboard & Data Explorer")
st.markdown("Explore sentence pairs and compare predictions from different sources.")

# --- Top-Level Dashboard ---
st.subheader("Dataset at a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sentence Pairs", f"{len(pairs_df):,}")
col2.metric("Total Predictions", f"{len(preds_df):,}")
col3.metric("🤖 Model Predictions", f"{model_count:,}")
col4.metric("🧠 Human Predictions", f"{human_count:,}")
st.markdown("---")

# --- Detailed View for Selected ID ---
if selected_id:
    # Find and display the sentence pair
    pair_info = pairs_df[pairs_df['sent_id'] == selected_id].iloc[0]

    st.subheader(f"📖 Details for `sent_id`: {selected_id}")
    with st.container(border=True):
        st.markdown(f"**Premise:** {pair_info['premise']}")
        st.markdown("---")
        st.markdown(f"**Hypothesis:** {pair_info['hypothesis']}")

    # Find and filter predictions for this sent_id
    relevant_preds = preds_df[
        (preds_df['sent_id'] == selected_id) &
        (preds_df['participant_type'].isin(selected_types))
        ]

    st.subheader("🔬 Predictions")
    if not relevant_preds.empty:
        for _, row in relevant_preds.iterrows():
            # Use our custom CSS class for a card layout
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)

                # Card header with metadata
                c1, c2 = st.columns([3, 1])
                with c1:
                    icon = "🤖" if row['participant_type'] == 'model' else "🧠"
                    st.markdown(f"**{icon} Participant:** `{row['participant_id']}`")

                with c2:
                    # Apply color classes to labels
                    label_class = f"label-{row['label']}"
                    st.markdown(f"**Label:** <span class='{label_class}'>{row['label'].upper()}</span>",
                                unsafe_allow_html=True)

                # Card body with explanation
                st.markdown("**Explanation:**")
                st.write(row['explanation'])

                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("No predictions found for the selected participant types.")

else:
    st.info("Please select a `sent_id` from the sidebar to view its details.")