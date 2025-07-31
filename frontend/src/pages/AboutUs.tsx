import { ModeToggle } from "@/components/mode-toggle";
import { Footer } from "@/components/ui/footer";
import { Hexagon, Github, Twitter } from "lucide-react";
import { useNavigate, Link } from "react-router-dom";
import img2 from "@/img/img2.webp";

const AboutUs = () => {
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
              <Link to="/about-us" className="text-sm font-semibold text-primary">
                About Us
              </Link>
              <Link to="/our-product" className="text-sm font-semibold text-foreground hover:text-muted-foreground transition-colors">
                Our Product
              </Link>
            </div>
            <ModeToggle />
          </div>
        </nav>
      </header>

      {/* Content */}
      <div className="px-4 sm:px-6 py-8 sm:py-16">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-12 items-center">
            <div className="animate-fade-in order-2 lg:order-1">
              <h1 className="text-3xl sm:text-4xl font-bold text-foreground mb-6 sm:mb-8">
                About PneumoDetect AI
              </h1>
              <div className="prose prose-lg max-w-none text-muted-foreground space-y-4 sm:space-y-6">
                <p className="text-base sm:text-lg">
                  Pneumonia remains the leading infectious killer of children under five, causing over 740,000 deaths each year worldwide. PneumoDetect AI is our lightweight, browser‐accessible assistant designed to empower clinicians in low‐resource settings with instant, AI‐powered chest X‑ray analysis.
                </p>
                <p className="text-sm sm:text-base">
                  Launched in 2024, our mission is to bridge healthcare gaps by delivering real‐time diagnostics, risk stratification, and visual interpretability without the need for expensive hardware or specialist radiologists.
                </p>
                <p className="text-sm sm:text-base">
                  Built by a multidisciplinary team of ML engineers, software developers, and medical advisors, PneumoDetect AI has been tested on over 100,000 NIH and Kaggle chest X-ray images and consistently achieves {'>'} 90% accuracy, with sensitivity and specificity tuned to minimize false negatives.
                </p>
                <div className="mt-6 sm:mt-8">
                  <h2 className="text-xl sm:text-2xl font-semibold text-foreground mb-3 sm:mb-4">Our Core Values</h2>
                  <ul className="space-y-2 sm:space-y-3">
                    <li className="flex items-start gap-2 sm:gap-3">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-sm sm:text-base"><strong>Clinical Accuracy:</strong> We prioritize rigorous model evaluation to ensure reliable pneumonia detection, achieving greater than 90% test accuracy.</span>
                    </li>
                    <li className="flex items-start gap-2 sm:gap-3">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-sm sm:text-base"><strong>Interpretability:</strong> Every prediction includes a Grad‐CAM heatmap highlighting affected lung regions, so clinicians can see exactly why the model made its call.</span>
                    </li>
                    <li className="flex items-start gap-2 sm:gap-3">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-sm sm:text-base"><strong>Accessibility:</strong> Our web‐based UI runs on any device—no GPU required—and works in low‐bandwidth environments with image compression and retry logic.</span>
                    </li>
                    <li className="flex items-start gap-2 sm:gap-3">
                      <div className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                      <span className="text-sm sm:text-base"><strong>Security & Privacy:</strong> We store data in HIPAA‑compliant databases, with secure OAuth2 authentication and full audit logging of every upload and report generation.</span>
                    </li>
                  </ul>
                </div>
              </div>
            </div>
            <div className="animate-fade-in order-1 lg:order-2">
              <img 
                src={img2} 
                alt="Medical professional reviewing AI‑generated chest X‑ray report" 
                className="rounded-lg shadow-2xl w-full h-auto"
              />
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
            href: "https://twitter.com/yourhandle",
            label: "Twitter",
          },
          {
            icon: <Github className="h-5 w-5" />,
            href: "https://github.com/yourrepo",
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

export default AboutUs;
