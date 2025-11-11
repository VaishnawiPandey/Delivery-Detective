# # route_optimizer.py

# import pandas as pd
# import networkx as nx

# def build_route_graph(data_path="dri_by_location.csv"):
#     """Build a graph using weather and sentiment-based weights"""
#     df = pd.read_csv(data_path)

#     # Create a graph
#     G = nx.Graph()

#     for i, row in df.iterrows():
#         loc = int(row["location"])
#         weather = row["weather_severity_index"]
#         sentiment = row["sentiment_score"]

#         # Compute weight = distance proxy × weather × sentiment penalty
#         weight = abs(weather) * (1.2 - sentiment)
#         G.add_node(loc, sentiment=sentiment, weather=weather)
        
#         if i > 0:
#             G.add_edge(i-1, i, weight=weight)

#     return G

# def optimize_route(G, start=0, end=None):
#     """Find shortest route from start to end"""
#     if end is None:
#         end = max(G.nodes)
#     path = nx.shortest_path(G, source=start, target=end, weight="weight")
#     cost = nx.shortest_path_length(G, source=start, target=end, weight="weight")
#     return path, cost

# if __name__ == "__main__":
#     G = build_route_graph()
#     path, cost = optimize_route(G)
#     print("🚚 Optimized Route:", path)
#     print("📉 Total Cost:", round(cost, 3))

# route_optimizer.py
import numpy as np
import pandas as pd
import networkx as nx
import folium
from itertools import permutations

# --- Sample data (replace with your real locations and coordinates) ---
locations = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
}

# --- Distance matrix simulation (in km) ---
distance_matrix = np.array([
    [0, 1427, 1743],   # Delhi → [Delhi, Mumbai, Bangalore]
    [1427, 0, 984],    # Mumbai → ...
    [1743, 984, 0]
])

def optimize_route():
    """
    Simple brute-force route optimizer.
    Returns best route (list of location names) and total cost (distance).
    """
    n = len(locations)
    indices = range(n)
    best_route = None
    min_cost = float('inf')

    for perm in permutations(indices):
        cost = sum(distance_matrix[perm[i], perm[i+1]] for i in range(n - 1))
        if cost < min_cost:
            min_cost = cost
            best_route = perm

    best_route_names = [list(locations.keys())[i] for i in best_route]
    print(f"🚚 Optimized Route: {best_route_names}")
    print(f"📉 Total Cost: {min_cost:.2f}")

    return best_route_names, min_cost

def generate_route_map(route):
    """
    Creates an interactive Folium map showing the route.
    """
    coords = [locations[city] for city in route]
    start = coords[0]

    m = folium.Map(location=start, zoom_start=5)

    # Add markers
    for city, coord in zip(route, coords):
        folium.Marker(location=coord, popup=city, icon=folium.Icon(color="blue")).add_to(m)

    # Draw route lines
    folium.PolyLine(coords, color="red", weight=3, opacity=0.8).add_to(m)

    m.save("route_map.html")
    print("🗺️ Route map saved as route_map.html")
    return "route_map.html"

if __name__ == "__main__":
    best_route, cost = optimize_route()
    generate_route_map(best_route)
