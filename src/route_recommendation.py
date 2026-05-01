# ============================================================
# ROUTE RECOMMENDATION MODULE
# Here we will do the time safety balance trade-off
# Logic:
# - Images are manually assigned to demo routes.
# - Each route receives an average danger score and travel time.
# - The user directly chooses their safety-time preference.
# - The system recommends the best route based on those preferences.
# ============================================================

import re
import pandas as pd


# ============================================================
# 1. Utility: normalize image names
# ============================================================

def normalize_image_name(image_name):
    
    name = str(image_name).strip()

    # Remove spaces after underscores: input_image_ 1.png -> input_image_1.png
    name = re.sub(r"_\s+", "_", name)

    # Remove duplicate suffixes like " (2)" before file extension
    name = re.sub(r"\s+\(\d+\)(?=\.)", "", name)

    return name


# ============================================================
# 2. Define demo route setup
# ============================================================

def build_demo_route_mapping():
  
    route_mapping = {
        # Route A: fastest but riskier
        "input_image_3.png": "Route A",
        "input_image_5.png": "Route A",
        "input_image_6.png": "Route A",
        "input_image_8.png": "Route A",

        # Route B: balanced option
        "input_image_2.png": "Route B",
        "input_image_4.png": "Route B",
        "input_image_7.png": "Route B",

        # Route C: safest but too slow
        "input_image_1.png": "Route C",
    }

    return route_mapping


def build_demo_route_times():

    route_times = {
        "Route A": 18,
        "Route B": 21,
        "Route C": 38,
    }

    return route_times


def build_demo_route_context():

    route_context = {
        "Route A": {
            "long_low_traffic_km": 0.0,
            "description": "fastest urban route",
        },
        "Route B": {
            "long_low_traffic_km": 5.0,
            "description": "balanced route with a calmer section",
        },
        "Route C": {
            "long_low_traffic_km": 2.0,
            "description": "longer alternative route",
        },
    }

    return route_context


# ============================================================
# 3. Ask user preferences for demo
# ============================================================

def ask_user_preferences():
    """
    Asks the user for their safety-time preferences during the demo.

    The user decides:
    1. Minimum percentage risk reduction needed to change route.
    2. Maximum percentage extra travel time they are willing to accept.
    """

    print("\n================ USER PREFERENCES ================")
    print("We will compare safer routes against the fastest route.")
    print("Press Enter to use the default values shown in brackets.\n")

    # Default values for smoother demo execution
    default_min_risk_reduction_pct = 15.0
    default_max_extra_time_pct = 20.0

    min_risk_input = input(
        "Minimum risk reduction needed to change route (%) "
        f"[default: {default_min_risk_reduction_pct}]: "
    ).strip()

    max_time_input = input(
        "Maximum extra travel time you are willing to accept (%) "
        f"[default: {default_max_extra_time_pct}]: "
    ).strip()

    if min_risk_input == "":
        min_risk_reduction_pct = default_min_risk_reduction_pct
    else:
        min_risk_reduction_pct = float(min_risk_input)

    if max_time_input == "":
        max_extra_time_pct = default_max_extra_time_pct
    else:
        max_extra_time_pct = float(max_time_input)

    print("\nSelected user preferences:")
    print(f"- Minimum risk reduction: {min_risk_reduction_pct:.1f}%")
    print(f"- Maximum extra travel time: {max_extra_time_pct:.1f}%")

    return min_risk_reduction_pct, max_extra_time_pct


# ============================================================
# 4. Classify route danger
# ============================================================

def classify_route_danger(score):

    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    else:
        return "High"


# ============================================================
# 5. Calculate route-level scores
# ============================================================

def calculate_route_summary(results_df, route_mapping, route_times, route_context=None):
    """
    Groups image predictions into routes.

    For each route, it calculates:
    - number of images
    - average danger score
    - maximum danger score
    - average visual risk
    - average weather risk
    - average audio risk
    - total detected people, vehicles, and bikes
    - travel time
    - danger level
    - optional route context, such as long low-traffic sections
    """

    df = results_df.copy()

    if "image_name" not in df.columns:
        raise ValueError("results.csv must contain an 'image_name' column.")

    if "final_danger_score" not in df.columns:
        raise ValueError("results.csv must contain a 'final_danger_score' column.")

    #Normalize image names for safer matching
    df["normalized_image_name"] = df["image_name"].apply(normalize_image_name)

    #Assign route name
    df["route_name"] = df["normalized_image_name"].map(route_mapping)

    #Print warning if any images are not mapped
    unmapped_images = df[df["route_name"].isna()]["image_name"].tolist()

    if unmapped_images:
        print("\nWarning: some images were not included in any route:")
        for image_name in unmapped_images:
            print("-", image_name)

    #Remove images not included in route mapping
    df = df.dropna(subset=["route_name"])

    if df.empty:
        raise ValueError(
            "No images matched the route mapping. "
            "Check that the image names in build_demo_route_mapping() "
            "match the image_name column in outputs/results.csv."
        )


    optional_columns = {
        "weather_score": 0.0,
        "audio_score": 0.0,
        "num_people": 0,
        "num_vehicles": 0,
        "num_bikes": 0,
        "visual_risk_score": 0.0,
    }

    for column_name, default_value in optional_columns.items():
        if column_name not in df.columns:
            df[column_name] = default_value

    route_summary = (
        df.groupby("route_name")
        .agg(
            num_images=("image_name", "count"),
            avg_danger_score=("final_danger_score", "mean"),
            max_danger_score=("final_danger_score", "max"),
            avg_visual_risk=("visual_risk_score", "mean"),
            avg_weather_risk=("weather_score", "mean"),
            avg_audio_risk=("audio_score", "mean"),
            total_people=("num_people", "sum"),
            total_vehicles=("num_vehicles", "sum"),
            total_bikes=("num_bikes", "sum"),
        )
        .reset_index()
    )

    route_summary["travel_time_min"] = route_summary["route_name"].map(route_times)

    if route_summary["travel_time_min"].isna().any():
        raise ValueError(
            "Some routes do not have travel times. "
            "Check build_demo_route_times()."
        )

    route_summary["danger_level"] = route_summary["avg_danger_score"].apply(
        classify_route_danger
    )

    if route_context is None:
        route_context = {}

    route_summary["long_low_traffic_km"] = route_summary["route_name"].apply(
        lambda route: route_context.get(route, {}).get("long_low_traffic_km", 0.0)
    )

    route_summary["route_description"] = route_summary["route_name"].apply(
        lambda route: route_context.get(route, {}).get("description", "")
    )

    route_summary = route_summary.sort_values(
        by="travel_time_min",
        ascending=True
    ).reset_index(drop=True)

    return route_summary


# ============================================================
# 6. Recommend best route using user preferences
# ============================================================

def recommend_route(
    route_summary,
    min_risk_reduction_pct=15,
    max_extra_time_pct=20,
):

    summary = route_summary.copy()

    # Fastest route as baseline
    fastest_route = summary.loc[summary["travel_time_min"].idxmin()]

    fastest_route_name = fastest_route["route_name"]
    fastest_time = fastest_route["travel_time_min"]
    fastest_danger = fastest_route["avg_danger_score"]

    recommendations = []

    for _, route in summary.iterrows():
        route_name = route["route_name"]
        route_time = route["travel_time_min"]
        route_danger = route["avg_danger_score"]
        route_max_danger = route["max_danger_score"]

        # Time comparison against fastest route
        extra_time = route_time - fastest_time
        extra_time_pct = (extra_time / fastest_time) * 100 if fastest_time > 0 else 0.0

        # Risk comparison against fastest route
        risk_reduction = fastest_danger - route_danger
        risk_reduction_pct = (
            (risk_reduction / fastest_danger) * 100
            if fastest_danger > 0
            else 0.0
        )

        # Fastest route is always kept as the baseline option
        if route_name == fastest_route_name:
            accepted = True
            reason = "Fastest route baseline"

        else:
            # A slower route is accepted only if it satisfies both user preferences:
            # 1. enough risk reduction
            # 2. acceptable extra time
            if (
                risk_reduction_pct >= min_risk_reduction_pct
                and extra_time_pct <= max_extra_time_pct
            ):
                accepted = True
                reason = (
                    f"Accepted because it reduces risk by {risk_reduction_pct:.1f}% "
                    f"and only increases travel time by {extra_time_pct:.1f}%."
                )

            else:
                accepted = False

                if (
                    risk_reduction_pct < min_risk_reduction_pct
                    and extra_time_pct > max_extra_time_pct
                ):
                    reason = (
                        f"Rejected because the risk reduction is only "
                        f"{risk_reduction_pct:.1f}% and the extra travel time is "
                        f"{extra_time_pct:.1f}%."
                    )

                elif risk_reduction_pct < min_risk_reduction_pct:
                    reason = (
                        f"Rejected because the risk reduction is only "
                        f"{risk_reduction_pct:.1f}%, below the user's required "
                        f"{min_risk_reduction_pct:.1f}%."
                    )

                else:
                    reason = (
                        f"Rejected because the extra travel time is "
                        f"{extra_time_pct:.1f}%, above the user's limit of "
                        f"{max_extra_time_pct:.1f}%."
                    )

        recommendations.append({
            "route_name": route_name,
            "travel_time_min": route_time,
            "avg_danger_score": route_danger,
            "max_danger_score": route_max_danger,
            "danger_level": route["danger_level"],
            "extra_time_min": extra_time,
            "extra_time_pct": extra_time_pct,
            "risk_reduction_vs_fastest": risk_reduction,
            "risk_reduction_pct_vs_fastest": risk_reduction_pct,
            "accepted_candidate": accepted,
            "decision_reason": reason,
        })

    decision_df = pd.DataFrame(recommendations)

    accepted_routes = decision_df[decision_df["accepted_candidate"] == True]

    if accepted_routes.empty:
        # This should not normally happen because the fastest route is accepted.
        accepted_routes = decision_df.copy()

    # Among accepted routes, choose the safest one.
    # max_danger_score is only a background tie-breaker, not a user input.
    selected_route = (
        accepted_routes
        .sort_values(
            by=["avg_danger_score", "max_danger_score", "travel_time_min"],
            ascending=[True, True, True]
        )
        .iloc[0]
    )

    return selected_route, decision_df


# ============================================================
# 7. Generate dynamic user notification
# ============================================================

def generate_route_notification(selected_route, route_summary):
    """
    Extra note:
    We included that long low-traffic sections are treated as advisory notifications,
    not as direct danger-score penalties.
    """

    route_name = selected_route["route_name"]
    danger_score = selected_route["avg_danger_score"]
    danger_level = selected_route["danger_level"]
    travel_time = selected_route["travel_time_min"]

    route_info = route_summary[
        route_summary["route_name"] == route_name
    ].iloc[0]

    total_people = route_info.get("total_people", 0)
    total_vehicles = route_info.get("total_vehicles", 0)
    total_bikes = route_info.get("total_bikes", 0)

    avg_weather_risk = route_info.get("avg_weather_risk", 0.0)
    avg_visual_risk = route_info.get("avg_visual_risk", 0.0)
    max_danger_score = route_info.get("max_danger_score", 0.0)
    long_low_traffic_km = route_info.get("long_low_traffic_km", 0.0)

    warnings = []

    #Vehicle warning
    if total_vehicles >= 6:
        warnings.append("high vehicle density was detected along this route")
    elif total_vehicles >= 3:
        warnings.append("moderate vehicle density was detected along this route")

    #Pedestrian warning
    if total_people >= 3:
        warnings.append("several pedestrians were detected near the road")
    elif total_people >= 1:
        warnings.append("pedestrian activity was detected")

    #Bike / cyclist warning
    if total_bikes >= 2:
        warnings.append("multiple cyclists or bikes were detected")
    elif total_bikes >= 1:
        warnings.append("a cyclist or bike was detected")

    #Weather warning
    if avg_weather_risk >= 0.45:
        warnings.append("adverse weather conditions may affect driving safety")
    elif avg_weather_risk >= 0.20:
        warnings.append("changing weather conditions may affect visibility or road grip")

    #Visual scene warning
    if avg_visual_risk >= 0.65:
        warnings.append("the visual scene shows a high-risk road environment")

    #Dangerous segment warning, only as explanation/advice
    if max_danger_score >= 0.75:
        warnings.append("one section of the route shows unusually high risk")

    #Base message
    if danger_score >= 0.66:
        base_message = (
            f"{route_name} selected. Estimated time: {travel_time} min. "
            f"Warning: this route has a HIGH danger level "
            f"(average risk score: {danger_score:.3f}). "
        )

    elif danger_score >= 0.33:
        base_message = (
            f"{route_name} selected. Estimated time: {travel_time} min. "
            f"Moderate risk detected "
            f"(average risk score: {danger_score:.3f}). "
        )

    else:
        base_message = (
            f"{route_name} selected. Estimated time: {travel_time} min. "
            f"Low risk detected "
            f"(average risk score: {danger_score:.3f}). "
        )

    #Dynamic warning part
    if warnings:
        warning_text = "Main risk factors: " + "; ".join(warnings) + ". "
    else:
        warning_text = (
            "No major specific risk factor was detected, but normal driving "
            "precautions are still recommended. "
        )

    #Driving advice
    if danger_score >= 0.66:
        advice = "Reduce speed, increase following distance, and stay highly alert."
    elif danger_score >= 0.33:
        advice = "Please stay aware and adjust your driving to the surrounding conditions."
    else:
        advice = "Continue following normal driving precautions."

    #Low-traffic / fatigue advisory
    fatigue_advice = ""

    if long_low_traffic_km >= 4:
        fatigue_advice = (
            f" This route also includes approximately {long_low_traffic_km:.1f} km "
            f"of low-traffic road. Maintain attention during this section and "
            f"consider resting if you feel tired."
        )

    notification = base_message + warning_text + advice + fatigue_advice

    return notification


# ============================================================
# 8. Generate decision explanation
# ============================================================

def generate_decision_explanation(
    selected_route,
    decision_df,
    min_risk_reduction_pct=15,
    max_extra_time_pct=20,
):

    selected_name = selected_route["route_name"]

    selected_row = decision_df[
        decision_df["route_name"] == selected_name
    ].iloc[0]

    explanation = (
        f"The system recommends {selected_name} because it best matches the user's "
        f"safety-time preferences. The user required at least a "
        f"{min_risk_reduction_pct:.1f}% risk reduction to change from the fastest route "
        f"and was willing to accept up to {max_extra_time_pct:.1f}% extra travel time. "
        f"{selected_name} has an average danger score of "
        f"{selected_row['avg_danger_score']:.3f}, a maximum danger score of "
        f"{selected_row['max_danger_score']:.3f}, and an estimated travel time of "
        f"{selected_row['travel_time_min']} minutes. "
        f"Decision reason: {selected_row['decision_reason']}."
    )

    return explanation


# ============================================================
# 9. Full route recommendation pipeline
# ============================================================

def run_route_recommendation(
    results_csv_path,
    min_risk_reduction_pct=15,
    max_extra_time_pct=20,
):

    results_df = pd.read_csv(results_csv_path)

    route_mapping = build_demo_route_mapping()
    route_times = build_demo_route_times()
    route_context = build_demo_route_context()

    route_summary = calculate_route_summary(
        results_df=results_df,
        route_mapping=route_mapping,
        route_times=route_times,
        route_context=route_context,
    )

    selected_route, decision_df = recommend_route(
        route_summary=route_summary,
        min_risk_reduction_pct=min_risk_reduction_pct,
        max_extra_time_pct=max_extra_time_pct,
    )

    notification = generate_route_notification(
        selected_route=selected_route,
        route_summary=route_summary,
    )

    explanation = generate_decision_explanation(
        selected_route=selected_route,
        decision_df=decision_df,
        min_risk_reduction_pct=min_risk_reduction_pct,
        max_extra_time_pct=max_extra_time_pct,
    )

    print("\nDecision explanation:")
    print(explanation)

    return route_summary, decision_df, selected_route, notification


# ============================================================
# 10. Optional direct test / interactive demo
# ============================================================

if __name__ == "__main__":
    from config import RESULTS_CSV_PATH

    # Ask the user directly during the demo
    min_risk_reduction_pct, max_extra_time_pct = ask_user_preferences()

    route_summary, decision_df, selected_route, notification = run_route_recommendation(
        results_csv_path=RESULTS_CSV_PATH,
        min_risk_reduction_pct=min_risk_reduction_pct,
        max_extra_time_pct=max_extra_time_pct,
    )

    print("\n================ ROUTE SUMMARY ================")
    print(route_summary)

    print("\n================ ROUTE DECISION ================")
    print(decision_df)

    print("\n================ SELECTED ROUTE ================")
    print(selected_route)

    print("\n================ USER NOTIFICATION ================")
    print(notification)
