# Mathematical & Computational Sciences Portfolio

This portfolio showcases the work and expertise of a Master's student in Applied Mathematics (MANU - Modelling and Numerical Analysis) at the University of Montpellier, with a background from École 42 Paris.

## Portfolio Structure

### Website (`portfolio-website/`)
A professional portfolio website built with the StartBootstrap Resume template, customized for mathematical and computational sciences.

**Features:**
- Clean, professional design optimized for academic/research profiles
- Responsive layout that works on all devices
- Sections for education, skills, projects, and research interests
- Programming competency scales (e.g., C++: 8.5/10)
- Emphasis on mathematical and computational expertise

### Quarto Projects (`quarto-projects/`)
Technical documentation and project showcases using Quarto for reproducible research.

**Documents:**
- `euler-methods-analysis.qmd`: Comprehensive analysis of Discontinuous Galerkin methods
- `mathematical-foundations.qmd`: Theoretical foundations of numerical methods

## Getting Started

### Prerequisites
- Node.js (for the website)
- Python 3.8+ (for numerical examples)
- Quarto (for rendering technical documents)

### Setting Up the Website

1. Navigate to the portfolio website directory:
   ```bash
   cd portfolio-website
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open your browser to `http://localhost:3000`

### Building for Production

```bash
cd portfolio-website
npm run build
```

The built website will be in the `dist/` directory.

### Rendering Quarto Documents

1. Install Quarto from https://quarto.org/docs/get-started/

2. Render documents:
   ```bash
   cd quarto-projects
   quarto render euler-methods-analysis.qmd
   quarto render mathematical-foundations.qmd
   ```

## Customization Guide

### Personal Information
Edit `portfolio-website/src/pug/index.pug` to update:
- Your name (currently "Your Name")
- Contact information
- Profile image (`assets/img/profile.jpg`)
- Social media links

### Skills & Competencies
Modify the skills section to reflect your actual competencies:
- Programming language proficiency levels
- Mathematical expertise areas
- Software tools and frameworks

### Projects
Add your specific projects in the projects section:
- Research work
- Course projects
- Open source contributions
- Academic achievements

### Education Details
Update education section with:
- Specific dates
- Grades/honors if applicable
- Relevant coursework
- Thesis topics

## Technical Stack

### Website
- **Framework**: StartBootstrap Resume template
- **Build Tools**: Node.js, npm
- **Styling**: Bootstrap 5, SCSS
- **Template Engine**: Pug
- **Icons**: Font Awesome

### Documentation
- **Quarto**: For reproducible research documents
- **LaTeX**: For mathematical typesetting
- **Python**: For code examples and visualization
- **HTML**: Output format for web deployment

## Repository Integration

This portfolio is designed to work alongside your numerical methods repository, showcasing:

### C++ Implementation
- Modern C++20 features
- Template metaprogramming
- Eigen library for linear algebra
- CMake build system

### Mathematical Methods
- Discontinuous Galerkin schemes
- Entropy-stable methods
- High-order time integration
- Convergence analysis

### Scientific Computing
- Computational fluid dynamics
- Numerical analysis
- Algorithm optimization
- Performance analysis

## Deployment Options

### GitHub Pages
1. Build the website: `npm run build`
2. Push the `dist/` directory to a GitHub Pages branch
3. Configure GitHub Pages settings

### Netlify
1. Connect your repository to Netlify
2. Set build command: `cd portfolio-website && npm run build`
3. Set publish directory: `portfolio-website/dist`

### Traditional Web Hosting
Upload the contents of `portfolio-website/dist/` to your web server.

## Academic Focus Areas

This portfolio highlights expertise in:

- **Applied Mathematics**: Functional analysis, PDEs, numerical methods
- **Computational Science**: Scientific computing, algorithm development
- **Programming**: C++, Python, mathematical software development
- **Research**: Numerical analysis, computational fluid dynamics

## Career Objectives

Currently seeking a 6-month internship in:
- Computational mathematics
- Numerical analysis
- Scientific software development
- Mathematical modeling
- High-performance computing

## Contact & Links

- **GitHub**: [Your GitHub Profile]
- **LinkedIn**: [Your LinkedIn Profile]
- **Email**: your.email@example.com
- **University**: University of Montpellier - MANU Program

---

*This portfolio demonstrates the intersection of rigorous mathematical training, computational expertise, and practical software development skills gained through École 42 Paris and the University of Montpellier.*