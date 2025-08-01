// src/components/file-upload-section.tsx
"use client";
import React, { useState } from "react";
import { FileUpload } from "@/components/ui/file-upload"; // Assuming this component provides the file input
import { Button } from "@/components/ui/button";
import { Trash2, Upload, Loader2 } from "lucide-react"; // Added Loader2 for loading state

export function FileUploadSection() {
  const [files, setFiles] = useState<File[]>([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [summary, setSummary] = useState<string>("");
  // Updated prediction state to match backend response structure (confidence as string)
  // CRITICAL UPDATE: Added 'probability' to the prediction state
  // confidence is now a number (float)
  const [prediction, setPrediction] = useState<null | { label: string; probability: string; confidence: number }>(null);
  const [isLoadingDiagnosis, setIsLoadingDiagnosis] = useState(false); // New loading state for diagnosis/heatmap
  const [isLoadingSummary, setIsLoadingSummary] = useState(false); // Loading state for summary
  const [diagnosisError, setDiagnosisError] = useState<string | null>(null); // Specific error for diagnosis
  const [summaryError, setSummaryError] = useState<string | null>(null); // Specific error for summary

  const handleFileUpload = (newFiles: File[]) => {
    const imageFiles = newFiles.filter(file =>
      file.type.startsWith("image/") &&
      (file.type.includes("png") || file.type.includes("jpg") || file.type.includes("jpeg") || file.type.includes("gif"))
    );
    setFiles(imageFiles);
    setCurrentImageIndex(0);
    setHeatmapUrl(null); // Clear heatmap on new upload
    setSummary(""); // Clear summary on new upload
    setPrediction(null); // Clear prediction on new upload
    setDiagnosisError(null); // Clear errors
    setSummaryError(null);
  };

  // This function will now trigger the /predict_with_cam endpoint
  // and update both prediction and heatmap.
  const handleGetDiagnosisAndHeatmap = async () => {
    if (files.length === 0) {
      setDiagnosisError("Please upload an image first.");
      return;
    }

    setIsLoadingDiagnosis(true);
    setDiagnosisError(null); // Clear previous errors
    setPrediction(null); // Clear previous prediction
    setHeatmapUrl(null); // Clear previous heatmap
    setSummary(""); // Clear summary when new diagnosis is made

    const fileToUpload = files[currentImageIndex];
    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict_with_cam', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Update prediction state with all necessary fields
      setPrediction({
        label: data.prediction,
        probability: data.probability, // Store probability as returned by backend (string)
        confidence: parseFloat(data.confidence_score) // Parse confidence to float here
      });

      // Update heatmap URL
      if (data.grad_cam_image_base64 && data.grad_cam_image_format) {
        setHeatmapUrl(`data:${data.grad_cam_image_format};base64,${data.grad_cam_image_base64}`);
      } else {
        setHeatmapUrl(null); // No heatmap returned
        setDiagnosisError("Heatmap data not received from API.");
      }

    } catch (err: unknown) {
      console.error('Diagnosis/Heatmap API error:', err);
      if (err instanceof Error) {
        setDiagnosisError(err.message || 'An unexpected error occurred during diagnosis.');
      } else {
        setDiagnosisError('An unexpected error occurred during diagnosis.');
      }
    } finally {
      setIsLoadingDiagnosis(false);
    }
  };

  const handleGenerateSummary = async () => {
    // Ensure a prediction has been made before generating a summary
    if (!prediction) {
      setSummaryError("Please get a diagnosis first to generate a summary.");
      return;
    }

    setIsLoadingSummary(true);
    setSummaryError(null);
    setSummary(""); // Clear previous summary

    // --- DEBUGGING LOGS ---
    console.log("--- Sending to /summarize_diagnosis ---");
    console.log("Prediction Label:", prediction.label, typeof prediction.label);
    console.log("Prediction Probability (string from backend):", prediction.probability, typeof prediction.probability);
    console.log("Prediction Confidence (number from state):", prediction.confidence, typeof prediction.confidence);
    console.log("Parsed Probability (float):", parseFloat(prediction.probability), typeof parseFloat(prediction.probability));
    // No need to parse confidence here, it's already a number
    console.log("---------------------------------------");
    // --- END DEBUGGING LOGS ---

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prediction: prediction.label,
          probability: parseFloat(prediction.probability),
          confidence_score: prediction.confidence, // Send as number directly
          // findings: "Optional additional clinical findings can go here" 
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = `HTTP error! status: ${response.status}`;
        // CRITICAL FIX: Handle FastAPI's 422 detail array
        if (errorData && errorData.detail && Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((err: any) => `${err.loc.join('.')} - ${err.msg}`).join('; ');
        } else if (errorData && errorData.detail) {
          errorMessage = errorData.detail;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();
      setSummary(data.diagnosis_summary);

    } catch (err: unknown) {
      console.error('Summary API error:', err);
      if (err instanceof Error) {
        setSummaryError(err.message || 'An unexpected error occurred during summary generation.');
      } else {
        setSummaryError('An unexpected error occurred during summary generation.');
      }
      setSummary("Failed to generate summary. Please try again."); // Fallback summary
    } finally {
      setIsLoadingSummary(false);
    }
  };

  return (
    <section id="upload" className="py-8 sm:py-12 px-4 animate-fade-in">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Section Header */}
        <div className="text-center mb-2">
          <h2 className="text-xl sm:text-2xl lg:text-3xl font-bold text-foreground mb-2">Upload & Analyze X-ray</h2>
          <p className="text-sm sm:text-base text-muted-foreground max-w-xl mx-auto">
            Upload a chest X-ray, generate heatmaps, get diagnosis & AI summaries instantly.
          </p>
        </div>

        {/* Upload + Heatmap Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Panel */}
          <div className="flex flex-col justify-between bg-card border border-border rounded-lg p-4 space-y-4 min-h-[400px]">
            <div className="flex-1 border rounded-lg bg-muted flex items-center justify-center">
              {files.length === 0 ? (
                <FileUpload onChange={handleFileUpload} />
              ) : (
                <img
                  src={URL.createObjectURL(files[currentImageIndex])}
                  alt={files[currentImageIndex].name}
                  className="rounded-md w-full h-full object-contain"
                />
              )}
            </div>
            <div className="flex justify-between items-center">
              {files.length > 0 ? (
                <>
                  <p className="text-sm text-foreground truncate">
                    {files[currentImageIndex].name}
                  </p>
                  <Button onClick={() => {
                    setFiles([]);
                    setHeatmapUrl(null);
                    setSummary("");
                    setPrediction(null);
                    setDiagnosisError(null);
                    setSummaryError(null);
                  }} size="sm" variant="outline">
                    <Trash2 className="w-4 h-4 mr-1" /> Clear
                  </Button>
                </>
              ) : (
                <span className="text-xs text-muted-foreground">PNG, JPG, GIF supported</span>
              )}
            </div>
          </div>

          {/* Heatmap Panel */}
          <div className="flex flex-col justify-between bg-card border border-border rounded-lg p-4 space-y-4 min-h-[400px]">
            <div className="flex-1 border rounded-lg bg-muted flex items-center justify-center">
              {heatmapUrl ? (
                <img src={heatmapUrl} alt="Grad-CAM" className="rounded-md w-full h-full object-contain" />
              ) : (
                <p className="text-sm text-muted-foreground text-center px-2">
                  Heatmap preview will appear here
                </p>
              )}
            </div>
            <div className="flex flex-wrap gap-2 justify-between items-center">
              <Button
                disabled={files.length === 0 || isLoadingDiagnosis}
                onClick={handleGetDiagnosisAndHeatmap}
                className="flex-1"
              >
                {isLoadingDiagnosis ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  'Generate Heatmap'
                )}
              </Button>
              <Button
                disabled={files.length === 0 || isLoadingDiagnosis}
                onClick={handleGetDiagnosisAndHeatmap}
                variant="outline"
                className="flex-1"
              >
                {isLoadingDiagnosis ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Diagnosing...
                  </>
                ) : (
                  'Get Diagnosis'
                )}
              </Button>
            </div>
            {diagnosisError && (
              <p className="text-red-500 text-sm text-center mt-2">{diagnosisError}</p>
            )}
          </div>
        </div>

        {/* Diagnosis Result */}
        <div className="bg-card border border-border rounded-lg p-4 space-y-2">
          <h3 className="text-base sm:text-lg font-semibold text-foreground">Diagnosis Result</h3>
          {prediction ? (
            <p className="text-sm sm:text-base text-muted-foreground">
              <strong>Result:</strong>{" "}
              <span className={`font-semibold ${prediction.label === 'Pneumonia' ? 'text-red-500' : 'text-green-500'}`}>
                {prediction.label}
              </span>{" "}
              detected with probability of <span className="font-semibold text-primary">{prediction.probability}</span> and confidence of <span className="font-semibold text-primary">{prediction.confidence.toFixed(2)}%</span>.
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">No prediction yet. Click "Get Diagnosis" to fetch result.</p>
          )}
        </div>

        {/* LLM Summary */}
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-base sm:text-lg font-semibold text-foreground">AI Summary</h3>
            <Button
              disabled={!prediction || isLoadingSummary} // Disable if no prediction or already loading
              onClick={handleGenerateSummary}
              size="sm"
              variant="outline"
              className="gap-1"
            >
              {isLoadingSummary ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4" />
                  Generate Summary
                </>
              )}
            </Button>
          </div>
          <div className="text-sm text-muted-foreground whitespace-pre-line">
            {summary || "No summary generated yet. Get a diagnosis and then click 'Generate Summary'."}
          </div>
          {summaryError && (
            <p className="text-red-500 text-sm text-center mt-2">{summaryError}</p>
          )}
        </div>
      </div>
    </section>
  );
}
