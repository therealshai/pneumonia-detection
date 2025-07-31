import { HeroSection } from "@/components/hero-section";
import { FileUploadSection } from "@/components/file-upload-section";
import { FooterSection } from "@/components/footer-section";
import { ModeToggle } from "@/components/mode-toggle";
import { Hexagon } from "lucide-react";
import { Link } from "react-router-dom";

const Index = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleLogoClick = (e: React.MouseEvent) => {
    e.preventDefault();
    scrollToTop();
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
              <Link to="/our-product" className="text-sm font-semibold text-foreground hover:text-muted-foreground transition-colors">
                Our Product
              </Link>
            </div>
            <ModeToggle />
          </div>
        </nav>
      </header>
      
      <HeroSection />
      
      <FooterSection />
    </div>
  );
};

export default Index;
