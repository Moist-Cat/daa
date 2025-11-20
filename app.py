import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple, Set, Optional
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import warnings
from itertools import combinations
from functools import reduce
from operator import or_

warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Freelancer-Project Matching Analysis", page_icon="🤝", layout="wide"
)


class FreelancerMatchingAnalyzer:
    def __init__(self):
        self.results = []

    def generate_dataset(self, config: Dict):
        """Generate synthetic dataset based on configuration"""
        np.random.seed(config["random_seed"])

        skills_pool = [f"skill_{i:03d}" for i in range(config["num_skills"])]

        # Generate freelancers
        freelancers = []
        for fid in range(config["num_freelancers"]):
            num_skills = np.minimum(
                np.random.pareto(1.5) + 1, config["max_skills_per_freelancer"]
            ).astype(int)
            skills = np.random.choice(skills_pool, size=num_skills, replace=False)
            competencies = np.random.randint(1, 6, size=num_skills)
            wage = np.random.lognormal(3.5, 0.5) * (1 + 0.1 * np.mean(competencies))

            freelancers.append(
                {
                    "id": f"F{fid:05d}",
                    "skills": list(skills),
                    "competencies": list(competencies),
                    "wage": max(20, min(150, wage)),
                }
            )

        # Generate projects
        projects = []
        for pid in range(config["num_projects"]):
            num_skills = np.random.randint(1, config["max_skills_per_project"] + 1)
            required_skills = np.random.choice(
                skills_pool, size=num_skills, replace=False
            )
            requirements = np.random.randint(3, 6, size=num_skills)

            projects.append(
                {
                    "id": f"P{pid:05d}",
                    "required_skills": list(required_skills),
                    "required_competencies": list(requirements),
                }
            )

        return freelancers, projects

    def find_optimal_team_raw_bf(self, project, freelancers):
        """(Obsolete) BF solution for optimal team selection"""
        skills = sorted(project["required_skills"])
        n = len(skills)
        full_mask = (1 << n) - 1

        # Filter relevant freelancers and compute their skill masks
        relevant_freelancers = {}
        for freelancer in freelancers:
            mask = 0
            freelancer_skills_dict = dict(
                zip(freelancer["skills"], freelancer["competencies"])
            )
            for i, skill in enumerate(skills):
                req_level = project["required_competencies"][i]
                if (
                    skill in freelancer_skills_dict
                    and freelancer_skills_dict[skill] >= req_level
                ):
                    mask |= 1 << i
            if not mask:
                continue
            fid, wage = freelancer["id"], freelancer["wage"]
            if mask in relevant_freelancers:
                if wage < relevant_freelancers[mask][1]:
                    relevant_freelancers[mask] = (fid, wage)
            else:
                relevant_freelancers[mask] = (fid, wage)

        # check every combination
        # n*n*2^n
        minima = float("inf")
        candidates = []
        if reduce(or_, relevant_freelancers) != full_mask:
            return minima, []
        for i in range(1, len(relevant_freelancers) + 1):
            for comb in combinations(relevant_freelancers, i):
                if reduce(or_, comb) == full_mask:
                    total_wage = 0
                    fs = []
                    for mask in comb:
                        fid, wage = relevant_freelancers[mask]
                        total_wage += wage
                        fs.append(fid)
                    if total_wage < minima:
                        minima = total_wage
                        candidates = fs
        return minima, candidates

    def find_optimal_team_greedy(self, project, freelancers):
        """Greedy algorithm for team selection - faster but suboptimal"""
        skills = sorted(project["required_skills"])
        n = len(skills)

        # Map skills to indices and requirements
        skill_to_req = {
            skill: req for skill, req in zip(skills, project["required_competencies"])
        }
        uncovered_skills = set(skills)
        selected_team = []
        total_cost = 0

        # Precompute which skills each freelancer can cover
        freelancer_coverage = []
        for freelancer in freelancers:
            freelancer_skills_dict = dict(
                zip(freelancer["skills"], freelancer["competencies"])
            )
            covered_skills = set()

            for skill in uncovered_skills:
                if (
                    skill in freelancer_skills_dict
                    and freelancer_skills_dict[skill] >= skill_to_req[skill]
                ):
                    covered_skills.add(skill)

            if covered_skills:
                # Cost-effectiveness: cost per skill covered
                cost_effectiveness = freelancer["wage"] / len(covered_skills)
                freelancer_coverage.append(
                    {
                        "id": freelancer["id"],
                        "wage": freelancer["wage"],
                        "covered_skills": covered_skills,
                        "cost_effectiveness": cost_effectiveness,
                    }
                )

        # Sort by cost-effectiveness (lowest cost per skill first)
        freelancer_coverage.sort(key=lambda x: x["cost_effectiveness"])

        # Greedy selection: pick most cost-effective freelancers until all skills covered
        while uncovered_skills and freelancer_coverage:
            best_freelancer = None
            best_skill_coverage = 0
            best_index = -1

            # Find freelancer that covers the most uncovered skills
            for i, freelancer in enumerate(freelancer_coverage):
                current_coverage = len(
                    freelancer["covered_skills"].intersection(uncovered_skills)
                )
                if current_coverage > best_skill_coverage:
                    best_freelancer = freelancer
                    best_skill_coverage = current_coverage
                    best_index = i

            if best_freelancer is None or best_skill_coverage == 0:
                break  # No freelancer can cover remaining skills

            # Add this freelancer to team
            selected_team.append(best_freelancer["id"])
            total_cost += best_freelancer["wage"]
            uncovered_skills -= best_freelancer["covered_skills"]

            # Remove selected freelancer from consideration
            freelancer_coverage.pop(best_index)

            # Update coverage for remaining freelancers
            for freelancer in freelancer_coverage:
                freelancer["covered_skills"] = freelancer[
                    "covered_skills"
                ].intersection(uncovered_skills)

        if uncovered_skills:
            return float("inf"), []  # Failed to cover all skills

        return total_cost, selected_team

    def find_optimal_team(self, project, freelancers):
        """Optimized solution with early pruning using DP with bitmasking"""
        skills = sorted(project["required_skills"])
        n = len(skills)
        full_mask = (1 << n) - 1

        # Filter relevant freelancers and compute their skill masks
        relevant_freelancers = {}
        for freelancer in freelancers:
            mask = 0
            freelancer_skills_dict = dict(
                zip(freelancer["skills"], freelancer["competencies"])
            )
            for i, skill in enumerate(skills):
                req_level = project["required_competencies"][i]
                if (
                    skill in freelancer_skills_dict
                    and freelancer_skills_dict[skill] >= req_level
                ):
                    mask |= 1 << i
            if not mask:
                continue

            fid, wage = freelancer["id"], freelancer["wage"]
            if mask in relevant_freelancers:
                if wage < relevant_freelancers[mask][1]:
                    relevant_freelancers[mask] = (fid, wage)
            else:
                relevant_freelancers[mask] = (fid, wage)

        # Check if coverage is possible
        if not relevant_freelancers or reduce(lambda x, y: x | y, relevant_freelancers.keys()) != full_mask:
            return float("inf"), []

        # DP array: dp[mask] = (min_cost, freelancer_ids)
        dp = [None] * (full_mask + 1)
        dp[0] = (0, [])  # Base case: no skills covered, zero cost

        # Sort freelancers by wage to try cheaper ones first (helps with pruning)
        sorted_freelancers = sorted(relevant_freelancers.items(), key=lambda x: x[1][1])

        min_cost = float("inf")
        best_team = []

        for mask, (fid, wage) in sorted_freelancers:
            # Iterate through all current states in reverse to avoid reusing the same freelancer
            for current_mask in range(full_mask, -1, -1):
                if dp[current_mask] is None:
                    continue

                new_mask = current_mask | mask
                new_cost = dp[current_mask][0] + wage

                # Early pruning: if we already exceed the best known cost, skip
                if new_cost >= min_cost:
                    continue

                if dp[new_mask] is None or new_cost < dp[new_mask][0]:
                    dp[new_mask] = (new_cost, dp[current_mask][1] + [fid])

                    # Update best solution if we found full coverage
                    if new_mask == full_mask and new_cost < min_cost:
                        min_cost = new_cost
                        best_team = dp[new_mask][1]

        return (min_cost, best_team) if min_cost != float("inf") else (float("inf"), [])

    def find_optimal_team_hybrid(self, project, freelancers, bf_threshold=8):
        """Hybrid approach: use DP for small problems, greedy for large ones"""
        n_skills = len(project["required_skills"])

        if n_skills <= bf_threshold:
            # Use optimal DP algorithm for small problems
            cost, team = self.find_optimal_team(project, freelancers)
            algorithm = "BF (Optimal)"
        else:
            # Use greedy algorithm for large problems
            cost, team = self.find_optimal_team_greedy(project, freelancers)
            algorithm = "Greedy (Approximate)"

        return cost, team, algorithm

    def run_experiment(self, config: Dict):
        """Run comprehensive experiment with multiple trials and algorithm comparison"""
        all_results = []

        for trial in range(config["num_trials"]):
            # Generate new dataset for each trial
            config["random_seed"] = config["base_seed"] + trial
            freelancers, projects = self.generate_dataset(config)

            trial_results = []
            selected_projects = np.random.choice(
                projects,
                size=min(config["projects_per_trial"], len(projects)),
                replace=False,
            )

            for project in selected_projects:
                # Run both algorithms for comparison
                algorithms = []

                # DP Algorithm
                start_time = time.time()
                bf_cost, bf_team = self.find_optimal_team(project, freelancers)
                bf_time = time.time() - start_time
                algorithms.append(("BF", bf_cost, len(bf_team), bf_time))

                # Greedy Algorithm
                start_time = time.time()
                greedy_cost, greedy_team = self.find_optimal_team_greedy(
                    project, freelancers
                )
                greedy_time = time.time() - start_time
                algorithms.append(
                    ("Greedy", greedy_cost, len(greedy_team), greedy_time)
                )

                # Hybrid Algorithm
                start_time = time.time()
                hybrid_cost, hybrid_team, hybrid_algo = self.find_optimal_team_hybrid(
                    project, freelancers, config["bf_threshold"]
                )
                hybrid_time = time.time() - start_time
                algorithms.append(
                    ("Hybrid", hybrid_cost, len(hybrid_team), hybrid_time)
                )

                # Calculate optimality gap for greedy
                optimal_cost = bf_cost
                if round(optimal_cost, 2) > round(greedy_cost, 2):
                    breakpoint()
                if optimal_cost != float("inf") and greedy_cost != float("inf"):
                    greedy_gap = ((greedy_cost - optimal_cost) / optimal_cost) * 100
                else:
                    greedy_gap = 0

                for algo_name, cost, team_size, comp_time in algorithms:
                    result = {
                        "trial": trial,
                        "project_id": project["id"],
                        "algorithm": algo_name,
                        "num_required_skills": len(project["required_skills"]),
                        "computation_time": comp_time,
                        "optimal_cost": cost if cost != float("inf") else None,
                        "real_optimal_cost": optimal_cost
                        if optimal_cost != float("inf")
                        else None,
                        "team_size": team_size if team_size else 0,
                        "skills_coverage": len(project["required_skills"]),
                        "avg_requirement": np.mean(project["required_competencies"]),
                        "using_bf_threshold": algo_name == "Hybrid",
                    }

                    # Add algorithm-specific metrics
                    if algo_name == "Greedy":
                        result["optimality_gap_percent"] = greedy_gap
                    elif algo_name == "Hybrid":
                        result["chosen_algorithm"] = hybrid_algo
                        result["optimality_gap_percent"] = (
                            ((hybrid_cost - optimal_cost) / optimal_cost) * 100
                            if optimal_cost != float("inf")
                            and hybrid_cost != float("inf")
                            else 0
                        )
                    else:
                        result["optimality_gap_percent"] = 0

                    trial_results.append(result)

            all_results.extend(trial_results)

        return pd.DataFrame(all_results), freelancers, projects

    def compute_comprehensive_stats(self, results_df: pd.DataFrame):
        """Compute comprehensive statistics by algorithm"""
        stats = {}
        algorithms = results_df["algorithm"].unique()

        for algo in algorithms:
            algo_data = results_df[results_df["algorithm"] == algo]
            prefix = f"{algo.lower()}_"

            # Basic statistics
            stats[prefix + "mean_computation_time"] = algo_data[
                "computation_time"
            ].mean()
            stats[prefix + "std_computation_time"] = algo_data["computation_time"].std()
            stats[prefix + "mean_optimal_cost"] = algo_data["optimal_cost"].mean()
            stats[prefix + "std_optimal_cost"] = algo_data["optimal_cost"].std()
            stats[prefix + "mean_team_size"] = algo_data["team_size"].mean()
            stats[prefix + "success_rate"] = (
                round(algo_data["optimal_cost"].dropna(), 2)
                >= round(algo_data["real_optimal_cost"].dropna(), 2)
            ).mean()

            if "optimality_gap_percent" in algo_data.columns:
                stats[prefix + "mean_optimality_gap"] = algo_data[
                    "optimality_gap_percent"
                ].mean()
                stats[prefix + "max_optimality_gap"] = algo_data[
                    "optimality_gap_percent"
                ].max()

        # Cross-algorithm comparisons
        if "bf_mean_optimal_cost" in stats and "greedy_mean_optimal_cost" in stats:
            stats["greedy_cost_increase_percent"] = (
                (stats["greedy_mean_optimal_cost"] - stats["bf_mean_optimal_cost"])
                / stats["bf_mean_optimal_cost"]
                * 100
            )

        if (
            "bf_mean_computation_time" in stats
            and "greedy_mean_computation_time" in stats
        ):
            stats["greedy_speedup_factor"] = (
                stats["bf_mean_computation_time"]
                / stats["greedy_mean_computation_time"]
            )

        return stats


def main():
    st.title("🤝 Freelancer-Project Matching Algorithm Analysis")
    st.markdown(
        """
    This application tests optimal (BF) vs greedy algorithms for freelancer-team selection under various conditions.
    """
    )

    analyzer = FreelancerMatchingAnalyzer()

    # Sidebar for configuration
    st.sidebar.header("Experiment Configuration")

    config = {
        "num_freelancers": st.sidebar.slider(
            "Number of Freelancers", 100, 10000, 1000, 100
        ),
        "num_projects": st.sidebar.slider("Number of Projects", 10, 1000, 100, 10),
        "num_skills": st.sidebar.slider("Total Skills in Market", 50, 500, 200, 10),
        "max_skills_per_freelancer": st.sidebar.slider(
            "Max Skills per Freelancer", 1, 10, 5, 1
        ),
        "max_skills_per_project": st.sidebar.slider(
            "Max Skills per Project", 1, 20, 8, 1
        ),
        "num_trials": st.sidebar.slider("Number of Trials", 1, 20, 5, 1),
        "projects_per_trial": st.sidebar.slider("Projects per Trial", 1, 50, 10, 1),
        "bf_threshold": st.sidebar.slider(
            "BF Threshold (skills)",
            1,
            12,
            8,
            1,
            help="Use BF for problems with skills ≤ this threshold",
        ),
        "base_seed": st.sidebar.number_input("Random Seed Base", 42, 1000, 42),
    }

    # Run experiment button
    if st.sidebar.button("Run Comprehensive Analysis"):
        with st.spinner("Running experiments and generating visualizations..."):
            results_df, freelancers, projects = analyzer.run_experiment(config)
            stats = analyzer.compute_comprehensive_stats(results_df)

            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                [
                    "📊 Summary Statistics",
                    "⚡ Algorithm Comparison",
                    "⏱️ Performance Analysis",
                    "💰 Cost Analysis",
                    "👥 Team Composition",
                    "🎯 Hybrid Analysis",
                ]
            )

            with tab1:
                st.header("Experiment Summary")

                # Key metrics by algorithm
                st.subheader("Algorithm Performance Summary")

                algo_data = (
                    results_df.groupby("algorithm")
                    .agg(
                        {
                            "computation_time": ["mean", "std"],
                            "optimal_cost": ["mean", "std"],
                            "team_size": "mean",
                            "optimality_gap_percent": "mean",
                        }
                    )
                    .round(4)
                )

                st.dataframe(algo_data, use_container_width=True)

                # Quick comparison metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    if "greedy_speedup_factor" in stats:
                        st.metric(
                            "Greedy Speedup Factor",
                            f"{stats['greedy_speedup_factor']:.1f}x",
                        )
                    if "greedy_cost_increase_percent" in stats:
                        st.metric(
                            "Greedy Cost Increase",
                            f"{stats['greedy_cost_increase_percent']:.1f}%",
                        )

                with col2:
                    if "bf_success_rate" in stats:
                        st.metric(
                            "BF Success Rate", f"{stats['bf_success_rate']*100:.1f}%"
                        )
                    if "greedy_success_rate" in stats:
                        st.metric(
                            "Greedy Success Rate",
                            f"{stats['greedy_success_rate']*100:.1f}%",
                        )

                with col3:
                    if "hybrid_mean_computation_time" in stats:
                        st.metric(
                            "Hybrid Mean Time",
                            f"{stats['hybrid_mean_computation_time']:.4f}s",
                        )
                    if "hybrid_mean_optimality_gap" in stats:
                        st.metric(
                            "Hybrid Optimality Gap",
                            f"{stats['hybrid_mean_optimality_gap']:.2f}%",
                        )

            with tab2:
                st.header("Algorithm Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    # Computation time comparison
                    fig = px.box(
                        results_df,
                        x="algorithm",
                        y="computation_time",
                        title="Computation Time by Algorithm (Log Scale)",
                        log_y=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Success rate by algorithm
                    success_rates = (
                        results_df.groupby("algorithm")
                        .apply(lambda x: (x["optimal_cost"] != float("inf")).mean())
                        .reset_index()
                    )
                    success_rates.columns = ["algorithm", "success_rate"]

                    fig = px.bar(
                        success_rates,
                        x="algorithm",
                        y="success_rate",
                        title="Success Rate by Algorithm",
                        color="algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Cost comparison
                    valid_results = results_df[
                        results_df["optimal_cost"] != float("inf")
                    ]
                    fig = px.box(
                        valid_results,
                        x="algorithm",
                        y="optimal_cost",
                        title="Optimal Cost Distribution by Algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Optimality gap distribution
                    if "optimality_gap_percent" in results_df.columns:
                        gap_data = results_df[
                            results_df["algorithm"].isin(["Greedy", "Hybrid"])
                        ]
                        fig = px.box(
                            gap_data,
                            x="algorithm",
                            y="optimality_gap_percent",
                            title="Optimality Gap Distribution (%)",
                        )
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.header("Performance Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Time vs skill complexity by algorithm
                    fig = px.line(
                        results_df,
                        x="num_required_skills",
                        y="computation_time",
                        color="algorithm",
                        title="Computation Time vs Skill Complexity by Algorithm",
                        log_y=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Performance breakdown by skill count
                    skill_algo_performance = (
                        results_df.groupby(["num_required_skills", "algorithm"])[
                            "computation_time"
                        ]
                        .mean()
                        .reset_index()
                    )

                    fig = px.bar(
                        skill_algo_performance,
                        x="num_required_skills",
                        y="computation_time",
                        color="algorithm",
                        barmode="group",
                        title="Mean Computation Time by Skill Count and Algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Algorithm choice effectiveness
                st.subheader("Hybrid Algorithm Effectiveness")
                hybrid_decisions = results_df[
                    results_df["algorithm"] == "Hybrid"
                ].copy()
                hybrid_decisions["correct_choice"] = (
                    hybrid_decisions["num_required_skills"] <= config["bf_threshold"]
                )

                if len(hybrid_decisions) > 0:
                    choice_analysis = (
                        hybrid_decisions.groupby("chosen_algorithm")
                        .agg(
                            {
                                "computation_time": "mean",
                                "optimality_gap_percent": "mean",
                                "project_id": "count",
                            }
                        )
                        .round(4)
                    )

                    st.write("Hybrid Algorithm Choices:")
                    st.dataframe(choice_analysis, use_container_width=True)

            with tab4:
                st.header("Cost Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Cost distribution by algorithm
                    valid_costs = results_df[results_df["optimal_cost"] != float("inf")]
                    fig = px.histogram(
                        valid_costs,
                        x="optimal_cost",
                        color="algorithm",
                        facet_col="algorithm",
                        title="Cost Distribution by Algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Cost vs team size
                    fig = px.scatter(
                        valid_costs,
                        x="team_size",
                        y="optimal_cost",
                        color="algorithm",
                        title="Team Cost vs Team Size by Algorithm",
                        opacity=0.6,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Cost-effectiveness analysis
                st.subheader("Cost-Effectiveness Trade-off")

                tradeoff_data = (
                    valid_costs.groupby("algorithm")
                    .agg({"optimal_cost": "mean", "computation_time": "mean"})
                    .reset_index()
                )

                fig = px.scatter(
                    tradeoff_data,
                    x="computation_time",
                    y="optimal_cost",
                    size=[100, 100, 100],
                    text="algorithm",
                    title="Cost vs Time Trade-off (Lower-left is better)",
                    labels={
                        "computation_time": "Time (seconds)",
                        "optimal_cost": "Mean Cost",
                    },
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

            with tab5:
                st.header("Team Composition Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Team size distribution by algorithm
                    valid_teams = results_df[results_df["team_size"] > 0]
                    fig = px.box(
                        valid_teams,
                        x="algorithm",
                        y="team_size",
                        title="Team Size Distribution by Algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Team size vs skill requirements
                    fig = px.scatter(
                        valid_teams,
                        x="num_required_skills",
                        y="team_size",
                        color="algorithm",
                        trendline="lowess",
                        title="Team Size vs Required Skills by Algorithm",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with tab6:
                st.header("Hybrid Algorithm Analysis")

                st.subheader("BF Threshold Effectiveness")

                # Analyze performance around the threshold
                threshold = config["bf_threshold"]
                hybrid_results = results_df[results_df["algorithm"] == "Hybrid"]

                # Create bins around threshold
                hybrid_results["skill_bin"] = pd.cut(
                    hybrid_results["num_required_skills"],
                    bins=[0, threshold - 1, threshold, threshold + 1, 20],
                    labels=[
                        f"≤{threshold-1}",
                        f"{threshold}",
                        f"≥{threshold+1}",
                        "Large",
                    ],
                )

                threshold_analysis = (
                    hybrid_results.groupby("skill_bin")
                    .agg(
                        {
                            "computation_time": "mean",
                            "optimality_gap_percent": "mean",
                            "chosen_algorithm": lambda x: x.value_counts().to_dict(),
                        }
                    )
                    .round(4)
                )

                st.dataframe(threshold_analysis, use_container_width=True)

                # Recommendation based on analysis
                st.subheader("Configuration Recommendations")

                recommendations = []

                if (
                    "greedy_mean_optimality_gap" in stats
                    and stats["greedy_mean_optimality_gap"] < 5
                ):
                    recommendations.append(
                        "✅ Greedy algorithm provides good quality (gap < 5%) with much better speed"
                    )

                if (
                    "greedy_speedup_factor" in stats
                    and stats["greedy_speedup_factor"] > 10
                ):
                    recommendations.append(
                        f"✅ Greedy algorithm is {stats['greedy_speedup_factor']:.1f}x faster than BF"
                    )

                if "bf_threshold" in config:
                    bf_usage = len(
                        hybrid_results[
                            hybrid_results["num_required_skills"] <= threshold
                        ]
                    ) / len(hybrid_results)
                    recommendations.append(
                        f"✅ Current threshold uses BF for {bf_usage*100:.1f}% of problems"
                    )

                for rec in recommendations:
                    st.write(rec)

                # Export results
                st.subheader("Export Results")
                if st.button("Download Results CSV"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"algorithm_comparison_{config['num_freelancers']}f.csv",
                        mime="text/csv",
                    )

    else:
        # Show instructions before running
        st.info(
            """
        ## 🚀 Algorithm Comparison Analysis
        
        This analyzer compares three approaches:
        
        **1. BF Algorithm (Optimal)**
        - Guaranteed optimal solution
        - Exponential time complexity O(2^n)
        - Best for small problems (n ≤ threshold)
        
        **2. Greedy Algorithm (Approximate)**
        - Fast, polynomial time
        - Provides good approximate solutions
        - Optimality gap typically 5-15%
        
        **3. Hybrid Algorithm**
        - Automatically chooses between BF and Greedy
        - Uses BF for small problems, Greedy for large ones
        - Configurable threshold
        
        ### Key Insights:
        - Set the BF threshold based on your performance requirements
        - Monitor the optimality gap vs speed trade-off
        - Hybrid approach provides the best balance for real-world use
        """
        )

        # Show threshold guidance
        st.subheader("BF Threshold Guidance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Conservative (6-8)**")
            st.write("- High solution quality")
            st.write("- Moderate speed")
            st.write("- Good for cost-sensitive applications")

        with col2:
            st.write("**Balanced (8-10)**")
            st.write("- Good quality/speed balance")
            st.write("- Handles most real projects")
            st.write("- Recommended default")

        with col3:
            st.write("**Aggressive (10+)**")
            st.write("- Maximum speed")
            st.write("- Larger optimality gaps")
            st.write("- For time-critical applications")


if __name__ == "__main__":
    main()
