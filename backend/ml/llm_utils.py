import os
import httpx
import json
import pandas as pd
import numpy as np
import cv2 # For heatmap analysis

async def call_perplexity_api(prompt_text_for_llm: str) -> str:
    """
    Makes an asynchronous call to the Perplexity AI API.
    """
    apiUrl = "https://api.perplexity.ai/chat/completions"
    apiKey = os.getenv("PERPLEXITY_API_KEY")
    if not apiKey:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set. Please set it in your .env file.")

    headers = {
        "Authorization": f"Bearer {apiKey}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar-pro", # Using sonar-pro as requested
        "messages": [
            {"role": "system", "content": "You are a highly experienced medical AI assistant specializing in chest X-ray diagnostics. Your role is to synthesize findings from multiple AI models and provide a concise, professional, and actionable summary for a medical doctor. Always prioritize patient safety and recommend further clinical correlation when appropriate. Do not act as a doctor or provide direct medical advice. Always include a disclaimer about consulting a healthcare professional."},
            {"role": "user", "content": prompt_text_for_llm}
        ],
        "max_tokens": 300, # Increased max_tokens for more detailed explanations
        "temperature": 0.7 # A balanced temperature for creative yet factual responses
    }

    async with httpx.AsyncClient(timeout=60.0) as client: # Increased timeout
        response = await client.post(
            apiUrl,
            headers=headers,
            json=payload
        )

    response.raise_for_status() # Raises an HTTPStatusError for bad responses (4xx or 5xx)
    result = response.json()

    if result.get("choices") and len(result["choices"]) > 0 and \
       result["choices"][0].get("message") and result["choices"][0]["message"].get("content"):
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Unexpected Perplexity API response structure: {result}")
        raise ValueError("LLM failed to generate summary due to unexpected response structure.")

def describe_heatmap(heatmap: np.ndarray, threshold: float = 0.7) -> str:
    """
    Analyzes a Grad-CAM heatmap and provides a textual description of highlighted regions.
    Args:
        heatmap (np.ndarray): The normalized Grad-CAM heatmap (0-1).
        threshold (float): The intensity threshold to consider a region "highlighted".
    Returns:
        str: A textual description of the heatmap.
    """
    if heatmap is None or heatmap.size == 0:
        return "Grad-CAM heatmap could not be generated or was empty."

    # Get dimensions
    h, w = heatmap.shape
    
    # Find coordinates of highlighted regions
    highlighted_coords = np.argwhere(heatmap > threshold)

    if len(highlighted_coords) == 0:
        return "Grad-CAM heatmap shows no significantly highlighted regions above the threshold."

    # Determine general location
    y_coords = highlighted_coords[:, 0]
    x_coords = highlighted_coords[:, 1]

    # Calculate bounding box (min_y, max_y, min_x, max_x)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_x, max_x = np.min(x_coords), np.max(x_coords)

    # Determine quadrants/regions based on relative position
    vertical_pos = []
    if max_y < h / 2: # Mostly upper half
        vertical_pos.append("upper")
    if min_y > h / 2: # Mostly lower half
        vertical_pos.append("lower")
    if not vertical_pos: # Spans across middle or is centered
        vertical_pos.append("central")
    
    horizontal_pos = []
    if max_x < w / 2: # Mostly left half
        horizontal_pos.append("left")
    if min_x > w / 2: # Mostly right half
        horizontal_pos.append("right")
    if not horizontal_pos: # Spans across middle or is centered
        horizontal_pos.append("central")

    # Combine for description
    location_parts = []
    if "upper" in vertical_pos: location_parts.append("upper")
    if "central" in vertical_pos and len(vertical_pos) == 1: location_parts.append("central")
    if "lower" in vertical_pos: location_parts.append("lower")

    if "left" in horizontal_pos: location_parts.append("left")
    if "central" in horizontal_pos and len(horizontal_pos) == 1: location_parts.append("central")
    if "right" in horizontal_pos: location_parts.append("right")

    location_desc = ""
    if "upper" in location_parts and "lower" in location_parts:
        location_desc += "spanning vertically from upper to lower "
    elif "upper" in location_parts:
        location_desc += "in the upper "
    elif "lower" in location_parts:
        location_desc += "in the lower "
    
    if "left" in location_parts and "right" in location_parts:
        location_desc += "and horizontally from left to right "
    elif "left" in location_parts:
        location_desc += "left "
    elif "right" in location_parts:
        location_desc += "right "

    if not location_desc:
        location_desc = "central " # Default if no clear quadrant dominance

    # More granular description for specific quadrants
    if "upper" in vertical_pos and "left" in horizontal_pos: location_desc = "upper-left "
    elif "upper" in vertical_pos and "right" in horizontal_pos: location_desc = "upper-right "
    elif "lower" in vertical_pos and "left" in horizontal_pos: location_desc = "lower-left "
    elif "lower" in vertical_pos and "right" in horizontal_pos: location_desc = "lower-right "
    elif "central" in vertical_pos and "central" in horizontal_pos: location_desc = "central "

    location_desc += "lung field."

    # Estimate extent/intensity
    mean_intensity = np.mean(heatmap[heatmap > threshold])
    intensity_desc = "moderately intense"
    if mean_intensity > 0.85:
        intensity_desc = "highly intense"
    elif mean_intensity < 0.6:
        intensity_desc = "mildly intense"

    # Final description
    description = (
        f"Grad-CAM heatmap highlights a {intensity_desc} region primarily in the {location_desc} "
        f"This area corresponds to the model's focus when making its prediction."
    )
    return description


async def get_llm_diagnosis_summary(
    model_v1_prediction: str,
    model_v1_probability: float,
    model_high_recall_prediction: str,
    model_high_recall_probability: float,
    findings: str | None,
    pneumonia_indicators_df: pd.DataFrame | None,
    grad_cam_description: str | None = None # NEW: Grad-CAM description input
) -> str:
    """
    Constructs a detailed prompt for the LLM based on multi-model predictions, knowledge base,
    and Grad-CAM insights, then calls the LLM API to get a diagnostic summary.
    """
    
    # --- Constructing the Core AI Findings ---
    ai_findings_summary = (
        f"Based on a chest X-ray analysis:\n"
        f"- **Model V1 (Balanced - Optimized for F1-Score):** Predicted '{model_v1_prediction}' with a probability of {model_v1_probability:.2f}.\n"
        f"- **Model High Recall (Optimized for Sensitivity):** Predicted '{model_high_recall_prediction}' with a probability of {model_high_recall_probability:.2f}.\n"
    )

    # --- Synthesizing Model Agreement/Disagreement ---
    model_agreement_insight = ""
    if model_v1_prediction == 'Pneumonia' and model_high_recall_prediction == 'Pneumonia':
        model_agreement_insight = "Both AI models indicate the presence of pneumonia, suggesting a consistent finding."
    elif model_v1_prediction == 'Normal' and model_high_recall_prediction == 'Pneumonia':
        model_agreement_insight = "The balanced model (V1) indicated 'Normal', but the high-recall model flagged 'Pneumonia'. This discrepancy suggests a subtle finding or a potential early stage of pneumonia that warrants closer clinical correlation."
    elif model_v1_prediction == 'Pneumonia' and model_high_recall_prediction == 'Normal':
        model_agreement_insight = "The balanced model (V1) indicated 'Pneumonia', but the high-recall model indicated 'Normal'. This suggests the balanced model might be detecting a less typical presentation or there might be a false positive from V1. Clinical correlation is highly recommended."
    else: # Both are Normal
        model_agreement_insight = "Both AI models consistently indicate normal findings for pneumonia."

    # --- Integrating Knowledge Base (RAG-like) ---
    knowledge_context = ""
    if pneumonia_indicators_df is not None:
        if model_v1_prediction == 'Pneumonia' or model_high_recall_prediction == 'Pneumonia':
            # Retrieve relevant indicators for pneumonia
            relevant_indicators = pneumonia_indicators_df[pneumonia_indicators_df['Relevance'] >= 3] # Filter for highly relevant indicators
            if not relevant_indicators.empty:
                knowledge_context = "\n\n**Relevant Clinical Indicators for Pneumonia:**\n"
                for index, row in relevant_indicators.iterrows():
                    knowledge_context += f"- **{row['Indicator']}**: {row['Justification']}\n"
        else: # Both models predict Normal
            knowledge_context = "\n\n**Typical Characteristics of a Normal Chest X-ray:**\n- No evidence of acute infiltrates, consolidations, or pleural effusions.\n- Clear lung fields and normal cardiac silhouette.\n"

    # --- Incorporating User Findings ---
    user_findings_context = ""
    if findings:
        user_findings_context = f"\n\n**Patient-Reported Symptoms/Clinical Findings:** {findings}.\n"
        # Optional: Add logic here to check if user findings align with pneumonia indicators
        if pneumonia_indicators_df is not None and (model_v1_prediction == 'Pneumonia' or model_high_recall_prediction == 'Pneumonia'):
            pneumonia_symptoms = pneumonia_indicators_df[pneumonia_indicators_df['Type'] == 'Symptom']['Indicator'].tolist()
            matching_symptoms = [s for s in pneumonia_symptoms if s.lower() in findings.lower()]
            if matching_symptoms:
                user_findings_context += f" (Note: Patient's reported findings include symptoms often associated with pneumonia: {', '.join(matching_symptoms)})."
            else:
                user_findings_context += " (Note: Patient's reported findings do not strongly align with typical pneumonia symptoms from the knowledge base)."
        elif pneumonia_indicators_df is not None and model_v1_prediction == 'Normal' and model_high_recall_prediction == 'Normal':
             pneumonia_symptoms = pneumonia_indicators_df[pneumonia_indicators_df['Type'] == 'Symptom']['Indicator'].tolist()
             matching_symptoms = [s for s in pneumonia_symptoms if s.lower() in findings.lower()]
             if matching_symptoms:
                 user_findings_context += f" (Note: Patient's reported findings include symptoms often associated with pneumonia: {', '.join(matching_symptoms)}. However, AI models indicate Normal. Clinical correlation is highly advised.)"

    # --- Incorporating Grad-CAM Insight ---
    grad_cam_context = ""
    if grad_cam_description:
        grad_cam_context = f"\n\n**Visual Interpretability (Grad-CAM):** {grad_cam_description}\n"

    # --- Final Prompt Assembly ---
    full_prompt = (
        f"{ai_findings_summary}\n"
        f"{model_agreement_insight}\n"
        f"{grad_cam_context}" # NEW: Add Grad-CAM context here
        f"{knowledge_context}"
        f"{user_findings_context}"
        "\n\nPlease provide a concise (under 200 words), professional diagnostic summary for a medical doctor. "
        "Explain the reasoning based on the combined AI insights, visual interpretations, and relevant indicators. "
        "Conclude with a clear recommendation for next steps, always emphasizing that this is AI-generated decision support and not a substitute for professional medical advice or clinical judgment. "
        "Do not use phrases like 'As an AI model...' or 'Based on the provided context...'. Just give the summary directly."
    )

    summary = await call_perplexity_api(full_prompt)
    return summary
