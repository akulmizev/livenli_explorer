import streamlit as st
import pandas as pd
import plotly.express as px
import textwrap

# --- Page Configuration ---
st.set_page_config(
    page_title="LiveNLI Explanation Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

px.defaults.template = "simple_white"


@st.cache_data
def load_data():
    try:
        pairs_df = pd.read_csv('db/sentence_pairs.csv')
        preds_df = pd.read_csv('db/predictions.csv')
        preds_df.loc[preds_df['participant_type'] == 'human', 'participant'] = 'human'

        human_count = preds_df[preds_df['participant_type'] == 'human'].shape[0]
        model_count = preds_df[preds_df['participant_type'] != 'human'].shape[0]

        return pairs_df, preds_df, human_count, model_count
    except FileNotFoundError:
        st.error("Make sure `sentence_pairs.csv` and `predictions.csv` are in the same directory.")
        return None, None, 0, 0


pairs_df, preds_df, human_count, model_count = load_data()

if pairs_df is None:
    st.stop()

all_experiments = sorted(preds_df['experiment'].unique())
all_participants = sorted(preds_df['participant'].unique())

with st.sidebar:
    st.title(" NLI Explorer")
    st.markdown("---")

    st.header("1. Select a Sentence Pair")

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
            index=0
        )
    else:
        search_id = st.text_input("Enter a `sent_id` (e.g., 20274n):", "")
        if search_id:
            if search_id in all_ids:
                selected_id = search_id
            else:
                st.warning("ID not found. Please try another.")

    st.markdown("---")
    st.header("2. Filter Experiments")

    selected_experiments = st.multiselect(
        "Toggle predictions by experiment:",
        options=all_experiments,
        default=['human', 'model_mix_lo', 'model_mix_hi']
    )

    st.markdown("---")
    st.header("3. Filter Participants")

    selected_participants = st.multiselect(
        "Toggle predictions by participant:",
        options=all_participants,
        default=all_participants
    )

    st.markdown("---")
    st.header("4. Plotting Dimensions")

    dim_reduction_method = st.radio(
        "Choose a dimensionality reduction method:",
        ('t-SNE', 'UMAP')
    )

st.title("📊 NLI Dashboard & Data Explorer")
st.markdown("Explore sentence pairs and compare predictions from different sources.")

st.subheader("Dataset at a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sentence Pairs", f"{len(pairs_df):,}")
col2.metric("Total Predictions", f"{len(preds_df):,}")
col3.metric("🤖 Model Predictions", f"{model_count:,}")
col4.metric("🧠 Human Predictions", f"{human_count:,}")
st.markdown("---")

if selected_id:
    pair_info = pairs_df[pairs_df['sent_id'] == selected_id].iloc[0]

    st.subheader(f"📖 Details for item: `{selected_id}`")
    with st.container(border=True):
        st.markdown(f"**Premise:** {pair_info['premise']}")
        st.markdown("---")
        st.markdown(f"**Hypothesis:** {pair_info['hypothesis']}")

    relevant_preds = preds_df[
        (preds_df['sent_id'] == selected_id) &
        (preds_df['experiment'].isin(selected_experiments)) &
        (preds_df['participant'].isin(selected_participants))
        ].copy()

    relevant_preds['wrapped_explanation'] = relevant_preds['explanation'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=50))
    )

    # New section using st.columns to place plots side-by-side
    st.subheader("🖼️ Explanation Embeddings & Class Distribution")
    col1, col2 = st.columns(2)

    if not relevant_preds.empty:
        # Scatter plot for Explanation Embeddings (moved to the first column)
        with col1:
            title_prefix = "t-SNE" if dim_reduction_method == 't-SNE' else "UMAP"
            fig = px.scatter(
                relevant_preds,
                x='tsne_1' if dim_reduction_method == 't-SNE' else 'umap_1',
                y='tsne_2' if dim_reduction_method == 't-SNE' else 'umap_2',
                color="participant",
                symbol="class_simple",
                hover_data=['wrapped_explanation'],
                height=700,
                size_max=150,
                title=f"{title_prefix} Plot of Explanations for ID: {selected_id}",
            )
            fig.update_traces(marker=dict(size=10))
            fig.update_traces(
                hoverlabel=dict(
                    namelength=0,  # hides legend title
                ),
                hovertemplate="<b>Explanation:</b> %{customdata[0]}"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Bar plot for Class Distribution (new code in the second column)
        with col2:
            # Prepare data for the bar plot
            class_counts = relevant_preds.groupby(['participant', 'class_simple']).size().reset_index(name='count')

            # Create the bar chart
            bar_fig = px.bar(
                class_counts,
                x="class_simple",
                y="count",
                color="participant",
                barmode="stack",
                height=700,
                title="Class Distribution by Participant",
                labels={'class_simple': 'Predicted Class', 'count': 'Number of Predictions'}
            )

            st.plotly_chart(bar_fig, use_container_width=True)

    else:
        st.info(f"No predictions found for the selected experiments and participants to plot.")

    st.markdown("---")

    st.subheader("🔬 Predictions")
    if not relevant_preds.empty:
        sorted_preds = relevant_preds.sort_values('participant')

        for _, row in sorted_preds.iterrows():
            with st.container():
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                c1, c2 = st.columns([3, 1])
                with c1:
                    icon = "🤖" if row['participant_type'] == 'model' else "🧠"
                    st.markdown(f"**{icon} Participant:** `{row['participant']}`")
                with c2:
                    label_class = f"label-{row['class_simple']}"
                    st.markdown(f"**Label:** <span class='{label_class}'>{row['class_simple'].upper()}</span>",
                                unsafe_allow_html=True)
                st.markdown("**Explanation:**")
                st.write(row['explanation'])
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No predictions found for the selected experiments and participants.")

else:
    st.info("Please select a `sent_id` from the sidebar to view its details.")
