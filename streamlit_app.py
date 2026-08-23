"""Streamlit web interface for the emergent ecosystem simulation."""

import streamlit as st
import numpy as np
from pathlib import Path
from dataclasses import replace
import pandas as pd
import matplotlib.pyplot as plt

# Import simulation components
from config import DEFAULT_CONFIG, EXPERIMENT_PRESETS, SimulationConfig
from src.simulation.world import World
from src.simulation.decision import create_decision_system

# Set Streamlit page config
st.set_page_config(
    page_title="Emergent Ecosystems",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { max-width: 1200px; }
    .stMetric { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌍 Emergent Ecosystems: Evolved Neural NPCs vs. Hand-Coded Game AI")
st.markdown("""
A 2D artificial-life simulation where autonomous prey and predator NPCs compete using:
- **Neural Networks** (evolved via neuroevolution)
- **Utility-based AI** (hand-coded decision rules)
- **Finite-State Machines** (baseline controller)
""")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Decision mode
    decision_mode = st.selectbox(
        "Decision Policy",
        options=["neural", "utility", "fsm"],
        help="Switch between evolved neural networks, utility-based AI, or FSM baseline"
    )
    
    # Scenario
    scenario = st.selectbox(
        "Scenario",
        options=["baseline", "resource_scarcity", "predator_pressure"],
        help="Different environmental conditions"
    )
    
    # Simulation parameters
    st.subheader("Initial Population")
    initial_prey = st.slider("Prey", 10, 100, EXPERIMENT_PRESETS[scenario].initial_prey)
    initial_predators = st.slider("Predators", 1, 30, EXPERIMENT_PRESETS[scenario].initial_predators)
    initial_resources = st.slider("Resources", 10, 200, EXPERIMENT_PRESETS[scenario].initial_resources)
    
    st.subheader("Simulation Speed")
    ticks = st.slider("Ticks to run", 100, 5000, 1000, step=100)
    updates_per_frame = st.slider("Updates per frame", 1, 10, 2)
    
    seed_val = st.number_input("Random seed", value=42, min_value=0)
    
    run_simulation = st.button("▶️ Run Simulation", use_container_width=True)

# Main area
if run_simulation:
    # Create config
    config = EXPERIMENT_PRESETS[scenario]
    config = replace(
        config,
        decision_mode=decision_mode,
        initial_prey=initial_prey,
        initial_predators=initial_predators,
        initial_resources=initial_resources,
        seed=seed_val,
        updates_per_frame=updates_per_frame,
    )
    
    # Run simulation
    with st.spinner(f"Running {ticks} ticks with {decision_mode} policy..."):
        world = World(config)
        world.populate_initial()
        
        # Storage for metrics
        metrics_history = {
            "tick": [],
            "prey_count": [],
            "predator_count": [],
            "resource_count": [],
            "avg_prey_energy": [],
            "avg_predator_energy": [],
        }
        
        # Run ticks
        progress_bar = st.progress(0)
        for tick in range(ticks):
            world.update(config.dt)
            
            # Collect metrics every 10 ticks
            if tick % 10 == 0 or tick == ticks - 1:
                metrics = world.metrics.get_latest_stats()
                metrics_history["tick"].append(tick)
                metrics_history["prey_count"].append(metrics.get("prey_count", 0))
                metrics_history["predator_count"].append(metrics.get("predator_count", 0))
                metrics_history["resource_count"].append(metrics.get("resource_count", 0))
                metrics_history["avg_prey_energy"].append(metrics.get("avg_prey_energy", 0))
                metrics_history["avg_predator_energy"].append(metrics.get("avg_predator_energy", 0))
            
            progress_bar.progress((tick + 1) / ticks)
    
    # Display results
    st.success(f"✅ Simulation complete!")
    
    # Metrics overview (3 columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Prey", metrics_history["prey_count"][-1])
    with col2:
        st.metric("Final Predators", metrics_history["predator_count"][-1])
    with col3:
        st.metric("Final Resources", metrics_history["resource_count"][-1])
    
    # Plots
    st.subheader("📊 Population Over Time")
    
    df = pd.DataFrame(metrics_history)
    
    # Population chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["tick"], df["prey_count"], label="Prey", color="#46A5FF", linewidth=2)
    ax.plot(df["tick"], df["predator_count"], label="Predators", color="#EB534C", linewidth=2)
    ax.plot(df["tick"], df["resource_count"], label="Resources", color="#58D26A", linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Count")
    ax.set_title(f"Population Dynamics - {scenario.replace('_', ' ').title()} ({decision_mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Energy chart
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["tick"], df["avg_prey_energy"], label="Avg Prey Energy", color="#46A5FF", linewidth=2)
    ax.plot(df["tick"], df["avg_predator_energy"], label="Avg Predator Energy", color="#EB534C", linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Energy")
    ax.set_title("Average Agent Energy Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Download CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download metrics as CSV",
        data=csv,
        file_name=f"emergent_ecosystem_{scenario}_{decision_mode}.csv",
        mime="text/csv"
    )

else:
    # Welcome screen
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Choose a policy** (Neural, Utility, or FSM) from the sidebar
        2. **Select a scenario** (baseline, scarcity, or predator pressure)
        3. **Adjust initial populations** as desired
        4. **Click "Run Simulation"** to watch the ecosystem evolve
        
        ### What are the policies?
        - **Neural**: Agents use evolved neural networks (neuroevolution, no backprop)
        - **Utility**: Hand-coded decision rules with scoring
        - **FSM**: Finite-state machine baseline controller
        
        All three policies share the same perception, actions, and physiology — **only the brain differs**.
        """)
    
    with col2:
        st.subheader("📚 Resources")
        st.markdown("""
        - [GitHub Repository](https://github.com/theapuhasan/emergent-ecosystem)
        - [Run Tests](https://github.com/theapuhasan/emergent-ecosystem#running-tests)
        - [Local Setup](https://github.com/theapuhasan/emergent-ecosystem#installation)
        """)
    
    st.info("""
    **💡 Tip**: Start with the "baseline" scenario on "utility" mode to see stable populations, 
    then experiment with "neural" mode to see how evolved controllers perform!
    """)

st.markdown("---")
st.caption("Emergent Ecosystems © 2024 | A study in artificial life and game AI")
