# # # app.py

# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import networkx as nx

# # from route_optimizer import build_route_graph, optimize_route
# # from rag_engine import query_index

# # from route_optimizer import optimize_route, generate_route_map



# # st.set_page_config(page_title="Delivery Detective Dashboard", layout="wide")

# # st.title("🕵️‍♂️ Delivery Detective - AI-Powered E-Commerce Optimization")

# # # Sidebar navigation
# # page = st.sidebar.radio("Navigate", ["📊 DRI Insights", "🔍 RAG Engine", "🚚 Route Optimizer"])

# # # --- PAGE 1: DRI INSIGHTS ---
# # if page == "📊 DRI Insights":
# #     st.header("📊 DRI Insights & Explainability")
# #     dri_df = pd.read_csv("dri_by_location.csv")
# #     st.dataframe(dri_df.head())

# #     st.subheader("Feature Importance")
# #     feat_df = pd.read_csv("dri_feature_importance.csv")
# #     st.bar_chart(feat_df.set_index("feature")["importance"])

# #     st.image("dri_shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)


# # # --- PAGE 2: RAG ENGINE ---
# # elif page == "🔍 RAG Engine":
# #     st.header("🔍 Ask Questions to Delivery Detective")

# #     query = st.text_input("Enter your question:")
# #     if st.button("Search"):
# #         response = query_index(query)
# #         st.write("### 🧾 Retrieved Insights")
# #         st.write(response)


# # # --- PAGE 3: ROUTE OPTIMIZER ---
# # # elif page == "🚚 Route Optimizer":
# # #     st.header("🚚 Dynamic Route Optimization")

# # #     G = build_route_graph("dri_by_location.csv")
# # #     nodes = list(G.nodes)
# # #     start = st.selectbox("Start Location", nodes, index=0)
# # #     end = st.selectbox("End Location", nodes, index=len(nodes)-1)

# # #     if st.button("Optimize Route"):
# # #         path, cost = optimize_route(G, start=start, end=end)
# # #         st.success(f"**Optimized Route:** {path}")
# # #         st.info(f"**Total Cost:** {round(cost, 3)}")
# # elif page == "🚚 Route Optimizer":
# #     st.header("🚚 Delivery Route Optimization")
    
# #     if st.button("Optimize Route"):
# #         best_route, total_cost = optimize_route()
# #         st.success(f"Optimized Route: {' → '.join(best_route)}")
# #         st.info(f"Total Distance: {total_cost:.2f} km")

# #         # Generate and show map
# #         map_path = generate_route_map(best_route)
# #         st.components.v1.html(open(map_path, 'r').read(), height=500)



# #         # Draw route graph
# #         fig, ax = plt.subplots()
# #         pos = nx.spring_layout(G, seed=42)
# #         nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, ax=ax)
# #         nx.draw_networkx_edges(G, pos, edgelist=list(zip(path, path[1:])), width=3, edge_color="orange")
# #         st.pyplot(fig)


# import streamlit as st
# import networkx as nx
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

# # ----------------------------
# # DATA (Replace this with your own distance matrix)
# # ----------------------------
# places = ["Warehouse", "Delhi", "Gurgaon", "Noida"]
# distance_matrix = [
#     [0, 7.2, 12.5, 10.1],
#     [7.2, 0, 8.4, 11.3],
#     [12.5, 8.4, 0, 6.8],
#     [10.1, 11.3, 6.8, 0],
# ]

# # Coordinates for map visualization (latitude, longitude)
# coords = {
#     "Warehouse": [28.7041, 77.1025],
#     "Delhi": [28.6139, 77.2090],
#     "Gurgaon": [28.4595, 77.0266],
#     "Noida": [28.5355, 77.3910],
# }

# # ----------------------------
# # HELPER FUNCTION
# # ----------------------------
# def nearest_neighbor_route(distance_matrix):
#     n = len(distance_matrix)
#     unvisited = set(range(1, n))
#     route = [0]  # Start at Warehouse
#     total_cost = 0

#     while unvisited:
#         last = route[-1]
#         next_city = min(unvisited, key=lambda x: distance_matrix[last][x])
#         total_cost += distance_matrix[last][next_city]
#         route.append(next_city)
#         unvisited.remove(next_city)

#     total_cost += distance_matrix[route[-1]][route[0]]  # Return to start
#     route.append(0)
#     return route, total_cost

# # ----------------------------
# # STREAMLIT UI
# # ----------------------------
# st.title("🚚 Delivery Detective - Route Optimization")
# st.write("AI-powered smart delivery route planner with interactive map visualization.")

# if st.button("Optimize Route"):
#     # Compute optimized route
#     route, total_cost = nearest_neighbor_route(distance_matrix)
#     named_route = [places[i] for i in route]

#     st.subheader("📍 Optimized Route:")
#     st.success(" → ".join(named_route))
#     st.metric("📉 Total Cost (approx. distance)", f"{total_cost:.2f} km")

#     # ----------------------------
#     # PLOT 1: Network Graph
#     # ----------------------------
#     st.subheader("📊 Route Network Visualization")
#     G = nx.Graph()

#     for i, src in enumerate(places):
#         for j, dest in enumerate(places):
#             if i != j:
#                 G.add_edge(src, dest, weight=distance_matrix[i][j])

#     fig, ax = plt.subplots()
#     pos = nx.spring_layout(G, seed=42)
#     nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, ax=ax)
#     nx.draw_networkx_edges(G, pos, edgelist=list(zip(named_route, named_route[1:])), width=3, edge_color="orange")
#     st.pyplot(fig)

#     # ----------------------------
#     # PLOT 2: Map Visualization
#     # ----------------------------
#     st.subheader("🗺️ Real Map Route")
#     lats = [coords[loc][0] for loc in named_route]
#     lons = [coords[loc][1] for loc in named_route]

#     fig_map = go.Figure()
#     fig_map.add_trace(go.Scattermapbox(
#         lat=lats,
#         lon=lons,
#         mode='markers+lines',
#         marker=dict(size=12, color='orange'),
#         text=named_route,
#         hoverinfo='text'
#     ))

#     fig_map.update_layout(
#         mapbox_style="open-street-map",
#         mapbox_zoom=9,
#         mapbox_center={"lat": 28.6, "lon": 77.2},
#         height=600
#     )

#     st.plotly_chart(fig_map)

# else:
#     st.info("Click the 'Optimize Route' button to start.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from rag_engine import query_index

# from route_optimizer import nearest_neighbor_route  # if you already have one
# from rag_engine import query_index   # uncomment when RAG engine is active

# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(page_title="Delivery Detective Dashboard", layout="wide")
st.title("🕵️‍♂️ Delivery Detective - AI-Powered E-Commerce Optimization")

# --------------------------------
# Sidebar Navigation
# --------------------------------
page = st.sidebar.radio("Navigate", ["📊 DRI Insights", "🔍 RAG Engine", "🚚 Route Optimizer"])

# --------------------------------
# PAGE 1: DRI INSIGHTS
# --------------------------------
if page == "📊 DRI Insights":
    st.header("📊 DRI Insights & Explainability")
    dri_df = pd.read_csv("dri_by_location.csv")
    st.dataframe(dri_df.head())

    st.subheader("Feature Importance")
    feat_df = pd.read_csv("dri_feature_importance.csv")
    st.bar_chart(feat_df.set_index("feature")["importance"])

    st.image("dri_shap_summary.png", caption="SHAP Summary Plot", use_column_width=True)


# --------------------------------
# PAGE 2: RAG ENGINE
# --------------------------------
# --- PAGE 2: RAG ENGINE ---
elif page == "🔍 RAG Engine":
    st.header("🔍 Ask Questions to Delivery Detective")

    query = st.text_input("💬 Enter your question:")
    if st.button("Search"):
        try:
            results = query_index(query)

            st.subheader("🧾 Retrieved Insights")

            if isinstance(results, str):
                # In case your query_index() returns a text summary instead of structured results
                st.write(results)
            elif isinstance(results, list) and len(results) > 0:
                for i, res in enumerate(results, 1):
                    if isinstance(res, tuple) and len(res) >= 3:
                        doc_name, score, snippet = res[:3]
                        st.markdown(f"**{i}. 📄 {doc_name}** — Relevance: `{score:.3f}`")
                        st.write(snippet[:300] + "...")
                        st.markdown("---")
                    else:
                        st.write(res)
            else:
                st.info("No relevant insights found. Try rephrasing your question.")

        except Exception as e:
            st.error(f"⚠️ Error while fetching results: {e}")



# --------------------------------
# PAGE 3: ROUTE OPTIMIZER
# --------------------------------
elif page == "🚚 Route Optimizer":
    st.header("🚚 Delivery Route Optimization")

    # ---- Define Data ----
    places = ["Warehouse", "Delhi", "Gurgaon", "Noida"]
    distance_matrix = [
        [0, 7.2, 12.5, 10.1],
        [7.2, 0, 8.4, 11.3],
        [12.5, 8.4, 0, 6.8],
        [10.1, 11.3, 6.8, 0],
    ]

    coords = {
        "Warehouse": [28.7041, 77.1025],
        "Delhi": [28.6139, 77.2090],
        "Gurgaon": [28.4595, 77.0266],
        "Noida": [28.5355, 77.3910],
    }

    # ---- Sidebar Inputs ----
    st.sidebar.markdown("### 🧭 Route Settings")
    from_city = st.sidebar.selectbox("Select Start Location", places, index=0)
    to_city = st.sidebar.selectbox("Select End Location", places, index=len(places)-1)

    # ---- Route Optimization ----
    if st.button("Optimize Route"):
        # Perform nearest neighbor optimization
        n = len(places)
        start_idx = places.index(from_city)
        end_idx = places.index(to_city)

        # Simple nearest neighbor starting from chosen city
        unvisited = set(range(n))
        unvisited.remove(start_idx)
        route = [start_idx]
        total_cost = 0

        while unvisited:
            last = route[-1]
            next_city = min(unvisited, key=lambda x: distance_matrix[last][x])
            total_cost += distance_matrix[last][next_city]
            route.append(next_city)
            unvisited.remove(next_city)

        # If end city is specified and not already last, adjust
        if route[-1] != end_idx:
            route.append(end_idx)
            total_cost += distance_matrix[route[-2]][end_idx]

        route.append(start_idx)  # return to start
        named_route = [places[i] for i in route]

        # ---- Display Results ----
        st.success(f"Optimized Route: {' → '.join(named_route)}")
        st.metric("📉 Total Distance", f"{total_cost:.2f} km")

        # ---- Graph Visualization ----
        st.subheader("📊 Route Network Visualization")
        G = nx.Graph()
        for i, src in enumerate(places):
            for j, dest in enumerate(places):
                if i != j:
                    G.add_edge(src, dest, weight=distance_matrix[i][j])

        fig, ax = plt.subplots()
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=700, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(named_route, named_route[1:])), width=3, edge_color="orange")
        st.pyplot(fig)

        # ---- Map Visualization ----
        st.subheader("🗺️ Real Map Route")
        lats = [coords[loc][0] for loc in named_route]
        lons = [coords[loc][1] for loc in named_route]

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers+lines',
            marker=dict(size=12, color='orange'),
            text=named_route,
            hoverinfo='text'
        ))

        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=9,
            mapbox_center={"lat": 28.6, "lon": 77.2},
            height=600
        )

        st.plotly_chart(fig_map)

    else:
        st.info("Select start and end cities from sidebar, then click **Optimize Route** to begin.")
