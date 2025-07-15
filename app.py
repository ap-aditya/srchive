import streamlit as st
import os
import pandas as pd
from pinecone import Pinecone
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

st.set_page_config(
    page_title="Srchive - ArXiv Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_connections():
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("arxiv-search")
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        db = mongo_client["arxiv_db"]
        collection = db["papers"]
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return index, collection, model
    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()


load_dotenv()
index, collection, model = init_connections()


with st.sidebar:
    st.title("🔎 Srchive")
    st.markdown(
        """
        **A semantic search engine for the modern researcher.**
        - Search across 30,000+ AI/ML papers from ArXiv.
        - Use natural language and multiple queries (separated by semicolons).
        - Classic papers are auto-identified by citation counts.
        """
    )
    st.info(
        "Try: `self-attention in computer vision; advances in large language models`"
    )
    st.markdown("---")
    st.link_button("Visit Arxiv", "https://arxiv.org/")

st.title("Srchive - ArXiv Semantic Search Engine")
st.caption(
    "Find the most relevant and influential research papers using natural language."
)

search_input = st.text_input(
    "Search for papers (use a semicolon ';' to separate multiple queries)",
    placeholder="e.g., transformers in vision; diffusion models for image generation",
    label_visibility="collapsed",
)


def perform_search(queries):
    all_results = []
    seen_ids = set()

    for query in queries:
        try:
            query_vector = model.encode(query.strip()).tolist()
            pinecone_results = index.query(
                vector=query_vector, top_k=15, include_metadata=False
            )

            result_ids = [res["id"] for res in pinecone_results.get("matches", [])]
            id_to_score = {
                res["id"]: res["score"] for res in pinecone_results.get("matches", [])
            }

            if not result_ids:
                continue

            papers_metadata = list(collection.find({"_id": {"$in": result_ids}}))

            for meta in papers_metadata:
                paper_id = meta["_id"]
                if paper_id not in seen_ids:
                    all_results.append(
                        {
                            "id": paper_id,
                            "score": id_to_score.get(paper_id, 0),
                            "title": meta.get("title", "No Title"),
                            "summary": meta.get("summary", "No summary available."),
                            "authors": meta.get("authors", "No authors listed"),
                            "pdf_url": meta.get("pdf_url", "#"),
                            "type": meta.get("type", "recent"),
                        }
                    )
                    seen_ids.add(paper_id)
        except Exception as e:
            st.error(f"Error processing query '{query}': {e}")

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:25]


if st.button("🔍 Search", use_container_width=True, type="primary"):
    if search_input.strip():
        queries = [q.strip() for q in search_input.split(";") if q.strip()]
        with st.spinner("Searching across 30,000+ papers..."):
            search_results = perform_search(queries)

        st.session_state.search_results = search_results
    else:
        st.warning("Please enter a search query.")

if "search_results" in st.session_state and st.session_state.search_results:
    results = st.session_state.search_results
    st.success(f"Found **{len(results)}** relevant papers.")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        filter_options = st.multiselect(
            "Filter by paper type:",
            options=["Recent", "Classic"],
            default=["Recent", "Classic"],
        )
    with col2:
        sort_option = st.selectbox("Sort by:", ("Relevance Score", "Title"))

    filtered_results = [r for r in results if r["type"].capitalize() in filter_options]

    if sort_option == "Title":
        filtered_results.sort(key=lambda x: x["title"])
    else:
        filtered_results.sort(key=lambda x: x["score"], reverse=True)

    if not filtered_results:
        st.info("No results match your filter criteria. Try adjusting the filters.")
    else:
        for idx, result in enumerate(filtered_results, 1):
            with st.container(border=True):
                col_main, col_meta = st.columns((4, 1))

                with col_main:
                    st.subheader(f"{result['title']}")
                    st.caption(f"Authors: {result['authors']}")
                    with st.expander("📄 View Abstract"):
                        st.write(result["summary"])

                with col_meta:
                    st.metric("Relevance", f"{result['score']:.3f}")
                    badge_text = (
                        f"🏅 Classic" if result["type"] == "classic" else f"🆕 Recent"
                    )
                    st.markdown(f"**{badge_text}**")
                    st.link_button(
                        "Read PDF ↗️", result["pdf_url"], use_container_width=True
                    )

            st.write("")

elif "search_results" in st.session_state:
    st.info("No results found. Please try different or broader search terms.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Built with ❤️ using Streamlit, Pinecone, and MongoDB."
    "</div>",
    unsafe_allow_html=True,
)
