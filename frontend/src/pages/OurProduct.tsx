import { ModeToggle } from "@/components/mode-toggle";
import { Footer } from "@/components/ui/footer";
import { Hexagon, Github, Twitter, Zap, Shield, Users, BarChart } from "lucide-react";
import { useNavigate, Link } from "react-router-dom";
import { FileUploadSection } from "@/components/file-upload-section"; // Adjust path if needed
import img1 from "@/img/img1.jpg"; // Adjust path if needed

const OurProduct = () => {
  const navigate = useNavigate();

  const handleLogoClick = (e: React.MouseEvent) => {
    e.preventDefault();
    navigate('/');
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Sticky Header */}
      <header className="sticky top-0 z-50 bg-background/95 backdrop-blur-md border-b border-border">
        <nav className="flex items-center justify-between p-3 sm:p-4 lg:p-6 max-w-7xl mx-auto">
          <button 
            onClick={handleLogoClick}
            className="flex items-center gap-x-2 hover:opacity-80 transition-opacity"
          >
            <Hexagon className="h-6 w-6 sm:h-8 sm:w-8 text-primary" />
            <span className="font-bold text-lg sm:text-xl">PneumoDetect AI</span>
          </button>
          
          <div className="flex items-center gap-x-4 sm:gap-x-8">
            <div className="flex gap-x-4 sm:gap-x-8">
              <Link to="/about-us" className="text-sm font-semibold text-foreground hover:text-muted-foreground transition-colors">
                About Us
              </Link>
              <Link to="/our-product" className="text-sm font-semibold text-primary">
                Our Product
              </Link>
            </div>
            <ModeToggle />
          </div>
        </nav>
      </header>

      {/* Content */}
      <div className="px-4 sm:px-6 py-8 sm:py-16">
        <div className="max-w-6xl mx-auto space-y-12 sm:space-y-16">
          {/* Hero Section */}
          <div className="text-center animate-fade-in">
            <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-6 sm:mb-8">AI Assistant for Pneumonia Detection</h1>
            <p className="text-base sm:text-xl text-muted-foreground max-w-3xl mx-auto">
              PneumoDetect AI helps clinicians in low-resource settings instantly diagnose pneumonia from chest X-rays using state-of-the-art deep learning and explainable AI.
            </p>
          </div>

          {/* Product Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-12 items-center">
            <div className="animate-fade-in order-2 lg:order-1">
              <img 
                src={img1} 
                alt="Chest X-ray analysis" 
                className="rounded-lg shadow-2xl w-full h-auto"
              />
            </div>
            <div className="animate-fade-in order-1 lg:order-2">
              <h2 className="text-2xl sm:text-3xl font-semibold text-foreground mb-4 sm:mb-6">Empowering Frontline Healthcare</h2>
              <p className="text-sm sm:text-base text-muted-foreground mb-4 sm:mb-6">
                Designed with rural clinics and hospitals in mind, our system delivers real-time AI predictions, visual heatmaps (Grad-CAM), PDF reports, and medical summaries, all accessible via a lightweight web interface.
              </p>
              <div className="space-y-3 sm:space-y-4">
                <div className="flex items-center gap-2 sm:gap-3">
                  <Zap className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                  <span className="text-sm sm:text-base">Fast EfficientNet-based pneumonia prediction</span>
                </div>
                <div className="flex items-center gap-2 sm:gap-3">
                  <Shield className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                  <span className="text-sm sm:text-base">Secure data handling with role-based access</span>
                </div>
                <div className="flex items-center gap-2 sm:gap-3">
                  <Users className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                  <span className="text-sm sm:text-base">Heatmaps and clinical summaries for interpretability</span>
                </div>
                <div className="flex items-center gap-2 sm:gap-3">
                  <BarChart className="h-4 w-4 sm:h-5 sm:w-5 text-primary" />
                  <span className="text-sm sm:text-base">Risk stratification: Low, Moderate, High</span>
                </div>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="animate-fade-in">
            <h2 className="text-2xl sm:text-3xl font-semibold text-foreground mb-8 sm:mb-12 text-center">Key Capabilities</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
              <div className="p-4 sm:p-6 bg-card rounded-lg border border-border hover:shadow-lg transition-shadow">
                <Zap className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-3 sm:mb-4" />
                <h3 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">CNN-Based Inference</h3>
                <p className="text-sm sm:text-base text-muted-foreground">We use a fine-tuned EfficientNet model exported to ONNX for high-speed, accurate predictions.</p>
              </div>
              <div className="p-4 sm:p-6 bg-card rounded-lg border border-border hover:shadow-lg transition-shadow">
                <BarChart className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-3 sm:mb-4" />
                <h3 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">Visual Heatmaps</h3>
                <p className="text-sm sm:text-base text-muted-foreground">Grad-CAM overlays highlight suspicious regions on the X-ray image for transparency and trust.</p>
              </div>
              <div className="p-4 sm:p-6 bg-card rounded-lg border border-border hover:shadow-lg transition-shadow">
                <Users className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-3 sm:mb-4" />
                <h3 className="text-lg sm:text-xl font-semibold mb-2 sm:mb-3">LLM-Based Summary</h3>
                <p className="text-sm sm:text-base text-muted-foreground">A language model generates a short clinical summary to assist doctors in understanding the diagnosis.</p>
              </div>
            </div>
          </div>

          {/* Upload Section */}
          <div className="animate-fade-in mt-16 sm:mt-24">
            <h2 className="text-2xl sm:text-3xl font-semibold text-foreground mb-6 text-center">
              Try PneumoDetect AI
            </h2>
            <p className="text-base sm:text-lg text-muted-foreground text-center max-w-2xl mx-auto mb-10">
              Upload a chest X-ray (JPEG/PNG) to receive an AI-powered pneumonia diagnosis with a heatmap and summary. Optimized for low-bandwidth environments.
            </p>
            <div className="max-w-4xl mx-auto p-4 sm:p-6 lg:p-8 bg-muted rounded-lg shadow-md border border-border">
              <FileUploadSection />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <Footer
        logo={<Hexagon className="h-10 w-10" />}
        brandName="PneumoDetect AI"
        socialLinks={[
          {
            icon: <Twitter className="h-5 w-5" />,
            href: "https://twitter.com",
            label: "Twitter",
          },
          {
            icon: <Github className="h-5 w-5" />,
            href: "https://github.com",
            label: "GitHub",
          },
        ]}
        mainLinks={[
          { href: "/about-us", label: "About Us" },
          { href: "/our-product", label: "Our Product" },
          { href: "/contact", label: "Contact" },
        ]}
        legalLinks={[
          { href: "/privacy", label: "Privacy Policy" },
          { href: "/terms", label: "Terms of Service" },
        ]}
        copyright={{
          text: "© 2024 PneumoDetect AI",
          license: "All rights reserved",
        }}
      />
    </div>
  );
};

export default OurProduct;
