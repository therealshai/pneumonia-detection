import { HeroLanding } from "@/components/hero";
import type { HeroLandingProps } from "@/components/hero";

export const HeroSection = () => {
  const heroProps: HeroLandingProps = {
    title: "AI-Powered Pneumonia Detection",
    description:
      "Analyze chest X-rays instantly with state-of-the-art deep learning. Get Grad-CAM heatmaps, downloadable reports, and LLM-based medical summaries — optimized for slow networks and offline support.",
    
    announcementBanner: {
      text: "Now available in low-resource clinics 🚑",
      linkText: "See how",
      linkHref: "/our-product"
    },
    
    callToActions: [
      { 
        text: "Upload Now", 
        href: "/our-product", 
        variant: "primary" 
      },
      { 
        text: "Learn How It Works", 
        href: "/our-product", 
        variant: "secondary" 
      }
    ],
    
    titleSize: "large",
    gradientColors: {
      from: "oklch(0.7 0.2 280)",
      to: "oklch(0.55 0.25 340)"
    },
    
    className: "min-h-screen animate-fade-in"
  };

  return <HeroLanding {...heroProps} />;
};
