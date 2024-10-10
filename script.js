// Loading screen
const loadingScreen = document.getElementById('loading-screen');
const loadingText = document.getElementById('loading-text');
const matrixEffect = document.getElementById('matrix-effect');
const languages = ['SURYODAYA', 'सूर्योदय', 'सूर्योदय', 'ਸੂਰਜ ਦਾ ਚੜ੍ਹਨਾ', 'সূর্যোদয়', 'సూర్యోదయం', 'സൂര്യോദയം', 'सूर्योदय', 'સૂર્યોદય', 'ಸೂರ್ಯೋದಯ', 'சூரியோதயம்'];
let currentLangIndex = 0;

function changeLoadingText() {
    loadingText.textContent = languages[currentLangIndex];
    currentLangIndex = (currentLangIndex + 1) % languages.length;
}

const loadingInterval = setInterval(changeLoadingText, 500);

// Matrix effect
function createMatrixEffect() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    matrixEffect.appendChild(canvas);

    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    const fontSize = 10;
    const columns = canvas.width / fontSize;
    const drops = [];

    for (let i = 0; i < columns; i++) {
        drops[i] = 1;
    }

    function draw() {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#1e90ff';
        ctx.font = fontSize + 'px monospace';

        for (let i = 0; i < drops.length; i++) {
            const text = characters.charAt(Math.floor(Math.random() * characters.length));
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);

            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
    }

    setInterval(draw, 33);
}

createMatrixEffect();

// Ensure the loading screen disappears after a set time
setTimeout(() => {
    loadingScreen.style.opacity = '0';
    setTimeout(() => {
        loadingScreen.style.display = 'none';
        clearInterval(loadingInterval);
    }, 500);
}, 5000);

// Typing animation
const typingText = document.getElementById('typing-text');
const textToType = "I'm a passionate student and aspiring developer.";
let i = 0;

function typeWriter() {
    if (i < textToType.length) {
        typingText.innerHTML += textToType.charAt(i);
        i++;
        setTimeout(typeWriter, 50);
    }
}

// Scroll Progress Indicator
window.addEventListener('scroll', () => {
    const scrollProgress = document.getElementById('scroll-progress');
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const progress = (scrollTop / scrollHeight) * 100;
    scrollProgress.style.width = progress + '%';
});

// Skills section
const skillsData = {
    programming: ['Python', 'HTML/CSS', 'C++', 'C'],
    datascience: ['Natural Language Processing (NLP)', 'Machine Learning Algorithms', 'Predictive Analytics', 'Data Representation and Modeling', 'Statistical Analysis'],
    tools: ['TensorFlow', 'PyTorch', 'Scikit-learn', 'SQL', 'Jupyter Notebooks', 'Git'],
    software: ['Data Structures & Algorithms', 'Object-Oriented Programming', 'Software Project Management', 'Compiler Design'],
    research: ['Data Pipeline Optimization', 'Multimodal Data Integration', 'Comprehensive Literature Reviews'],
    collaboration: ['Team Collaboration', 'Agile Methodologies', 'Start-up Development']
};

const skillDetails = document.getElementById('skill-details');
const skillCategoryTitle = document.getElementById('skill-category-title');
const skillList = document.getElementById('skill-list');
const closeSkillDetails = document.getElementById('close-skill-details');

function showSkillDetails(category) {
    const skills = skillsData[category];
    skillCategoryTitle.textContent = document.querySelector(`.skill-hex[data-category="${category}"] h3`).textContent;
    skillList.innerHTML = skills.map(skill => `<li>${skill}</li>`).join('');
    skillDetails.style.display = 'block';
}

document.querySelectorAll('.skill-hex').forEach(hex => {
    hex.addEventListener('click', () => {
        showSkillDetails(hex.dataset.category);
    });
});

closeSkillDetails.addEventListener('click', () => {
    skillDetails.style.display = 'none';
});

// Close skill details when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === skillDetails) {
        skillDetails.style.display = 'none';
    }
});

// Floating Action Button
const floatingActionButton = document.getElementById('floating-action-button');
window.addEventListener('scroll', () => {
    if (window.pageYOffset > 300) {
        floatingActionButton.classList.add('visible');
    } else {
        floatingActionButton.classList.remove('visible');
    }
});

floatingActionButton.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
});

// Improved Smooth Scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        smoothScroll(this.getAttribute('href'), 1000);
    });
});

function smoothScroll(target, duration) {
    var targetElement = document.querySelector(target);
    var targetPosition = targetElement.getBoundingClientRect().top;
    var startPosition = window.pageYOffset;
    var distance = targetPosition - startPosition;
    var headerOffset = document.querySelector('header').offsetHeight;
    var startTime = null;

    function animation(currentTime) {
        if (startTime === null) startTime = currentTime;
        var timeElapsed = currentTime - startTime;
        var run = ease(timeElapsed, startPosition, distance - headerOffset, duration);
        window.scrollTo(0, run);
        if (timeElapsed < duration) requestAnimationFrame(animation);
    }

    function ease(t, b, c, d) {
        t /= d / 2;
        if (t < 1) return c / 2 * t * t + b;
        t--;
        return -c / 2 * (t * (t - 2) - 1) + b;
    }

    requestAnimationFrame(animation);
}

// Optimize scroll performance
let ticking = false;
window.addEventListener('scroll', () => {
    if (!ticking) {
        window.requestAnimationFrame(() => {
            // Update scroll-dependent features here
            updateScrollProgress();
            updateFloatingActionButton();
            revealElementsOnScroll();
            ticking = false;
        });
        ticking = true;
    }
});

function updateScrollProgress() {
    const scrollProgress = document.getElementById('scroll-progress');
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const progress = (scrollTop / scrollHeight) * 100;
    scrollProgress.style.width = progress + '%';
}

function updateFloatingActionButton() {
    const floatingActionButton = document.getElementById('floating-action-button');
    if (window.pageYOffset > 300) {
        floatingActionButton.classList.add('visible');
    } else {
        floatingActionButton.classList.remove('visible');
    }
}

function revealElementsOnScroll() {
    const revealElements = document.querySelectorAll('.reveal');
    revealElements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        if (elementTop < windowHeight - 100) {
            element.classList.add('active');
        }
    });
}

// Rotating text effect
class TxtRotate {
  constructor(el, toRotate, period) {
    this.toRotate = toRotate;
    this.el = el;
    this.loopNum = 0;
    this.period = parseInt(period, 10) || 2000;
    this.txt = '';
    this.tick();
    this.isDeleting = false;
  }
  tick() {
    let i = this.loopNum % this.toRotate.length;
    let fullTxt = this.toRotate[i];

    if (this.isDeleting) {
      this.txt = fullTxt.substring(0, this.txt.length - 1);
    } else {
      this.txt = fullTxt.substring(0, this.txt.length + 1);
    }

    this.el.innerHTML = '<span class="wrap">'+this.txt+'</span>';

    let that = this;
    let delta = 300 - Math.random() * 100;

    if (this.isDeleting) { delta /= 2; }

    if (!this.isDeleting && this.txt === fullTxt) {
      delta = this.period;
      this.isDeleting = true;
    } else if (this.isDeleting && this.txt === '') {
      this.isDeleting = false;
      this.loopNum++;
      delta = 500;
    }

    setTimeout(function() {
      that.tick();
    }, delta);
  }
}

window.onload = function() {
  let elements = document.getElementsByClassName('txt-rotate');
  for (let i=0; i<elements.length; i++) {
    let toRotate = elements[i].getAttribute('data-rotate');
    let period = elements[i].getAttribute('data-period');
    if (toRotate) {
      new TxtRotate(elements[i], JSON.parse(toRotate), period);
    }
  }

  // Particles.js configuration
  particlesJS('particles-js', {
    particles: {
      number: { value: 80, density: { enable: true, value_area: 800 } },
      color: { value: "#1e90ff" },
      shape: { type: "circle" },
      opacity: { value: 0.5, random: false },
      size: { value: 3, random: true },
      line_linked: { enable: true, distance: 150, color: "#1e90ff", opacity: 0.4, width: 1 },
      move: { enable: true, speed: 6, direction: "none", random: false, straight: false, out_mode: "out", bounce: false }
    },
    interactivity: {
      detect_on: "canvas",
      events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true },
      modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
    },
    retina_detect: true
  });
};

// Project modal functionality
const modal = document.getElementById('project-modal');
const modalTitle = document.getElementById('modal-title');
const modalDescription = document.getElementById('modal-description');
const modalLink = document.getElementById('modal-link');
const closeBtn = document.getElementsByClassName('close')[0];

const projectData = {
    kancha: {
        title: "Kancha VA",
        description: "Meet Kancha VA, my latest brainchild—a virtual assistant with a Nepali twist. Unlike the usual assistants, Kancha VA speaks your language (literally!). Designed to understand and solve problems in Nepali, this AI-powered helper caters to a wide audience, from those seeking quick translations to people needing smart solutions for everyday challenges. Think of it as your tech-savvy friend who happens to know everything… in Nepali.\n\nBuilt on cutting-edge large language models (LLMs), Kancha VA is all about breaking down language barriers and making AI feel more relatable. Whether you need a task done or just some friendly advice in Nepali, Kancha VA has your back.",
        link: "https://github.com/SirAlchemist1/Kancha-VA"
    },
    moodtune: {
        title: "Moodtune",
        description: "Ever wished for the perfect playlist at the tap of a button? Enter Moodtune, your personal playlist curator. Using simple prompts like \"shower music,\" Moodtune will instantly create a playlist to match your vibe, thanks to the magic of the Last.fm API and some LLM-powered smarts. Want to switch things up for a workout or a chill evening? Just tell Moodtune, and it's got the tunes ready.\n\nI've got even bigger plans for Moodtune—stay tuned for even more exciting features coming soon.",
        link: "https://github.com/SirAlchemist1/Mood-Tunes-Ai"
    }
};

document.querySelectorAll('.project-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const project = this.closest('.project-card').dataset.project;
        const data = projectData[project];
        modalTitle.textContent = data.title;
        modalDescription.textContent = data.description;
        modalLink.href = data.link;
        modal.style.display = "block";
    });
});

closeBtn.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

// Add this to your existing script.js file

document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles for home section
    particlesJS('home-particles', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#3498db" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: "#3498db", opacity: 0.4, width: 1 },
            move: { enable: true, speed: 2, direction: "none", random: false, straight: false, out_mode: "out", bounce: false }
        },
        interactivity: {
            detect_on: "canvas",
            events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true },
            modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
        },
        retina_detect: true
    });

    const timelineItems = document.querySelectorAll('.timeline-item');

    function isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    }

    function animateTimeline() {
        timelineItems.forEach(item => {
            if (isElementInViewport(item)) {
                item.classList.add('aos-animate');
            }
        });
    }

    // Initial check
    animateTimeline();

    // Check on scroll
    window.addEventListener('scroll', animateTimeline);

    const educationCards = document.querySelectorAll('.education-card');

    educationCards.forEach(card => {
        card.addEventListener('click', () => {
            card.querySelector('.card-inner').style.transform = 
                card.querySelector('.card-inner').style.transform === 'rotateY(180deg)' 
                    ? 'rotateY(0deg)' 
                    : 'rotateY(180deg)';
        });
    });

    // Intersection Observer for animation on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
            }
        });
    }, { threshold: 0.1 });

    educationCards.forEach(card => {
        observer.observe(card);
        card.style.animationPlayState = 'paused';
    });

    // Responsive Navigation
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    const navLinks = document.querySelectorAll('.nav-links li');

    burger.addEventListener('click', () => {
        nav.classList.toggle('active');
        burger.classList.toggle('active');

        // Animate links
        navLinks.forEach((link, index) => {
            if (link.style.animation) {
                link.style.animation = '';
            } else {
                link.style.animation = `navLinkFade 0.5s ease forwards ${index / 7 + 0.3}s`;
            }
        });
    });

    // Contact Form Submission
    const contactForm = document.getElementById('contact-form');
    contactForm.addEventListener('submit', function(e) {
        e.preventDefault();
        // Here you would typically send the form data to a server
        alert('Thank you for your message! I will get back to you soon.');
        contactForm.reset();
    });

    // Animate contact items on scroll
    const contactItems = document.querySelectorAll('.contact-item');
    const contactObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
            }
        });
    }, { threshold: 0.1 });

    contactItems.forEach(item => {
        contactObserver.observe(item);
        item.style.animationPlayState = 'paused';
    });

    // Animate timeline items on scroll (assuming this is the other observer causing the error)
    const timelineObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('aos-animate');
            }
        });
    }, { threshold: 0.1 });

    timelineItems.forEach(item => {
        timelineObserver.observe(item);
    });

    // Awards section animations
    const awardCards = document.querySelectorAll('.award-card');
    
    awardCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.querySelector('.award-icon').style.transform = 'scale(1.2)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.querySelector('.award-icon').style.transform = 'scale(1)';
        });
    });

    // Animate awards on scroll
    const awardObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('aos-animate');
            }
        });
    }, { threshold: 0.1 });

    awardCards.forEach(card => {
        awardObserver.observe(card);
    });
});

AOS.init({
    duration: 1000,
    once: true
});

// Add this to your existing script.js file

document.addEventListener('DOMContentLoaded', () => {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-links li');
    const burger = document.querySelector('.burger');
    const userTimeElement = document.getElementById('user-time');

    // Function to update time
    function updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        userTimeElement.textContent = timeString;
    }

    // Update time immediately and then every second
    updateTime();
    setInterval(updateTime, 1000);

    // Scroll effect for navbar
    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Burger menu toggle
    burger.addEventListener('click', () => {
        const nav = document.querySelector('.nav-links');
        nav.classList.toggle('nav-active');

        // Animate links
        navLinks.forEach((link, index) => {
            if (link.style.animation) {
                link.style.animation = '';
            } else {
                link.style.animation = `navLinkFade 0.5s ease forwards ${index / 7 + 0.3}s`;
            }
        });

        // Burger animation
        burger.classList.toggle('toggle');
    });

    // Close menu when a link is clicked
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const nav = document.querySelector('.nav-links');
            nav.classList.remove('nav-active');
            burger.classList.remove('toggle');
            navLinks.forEach(link => {
                link.style.animation = '';
            });
        });
    });
});

document.addEventListener('DOMContentLoaded', () => {
    const homeSection = document.getElementById('home');
    
    // Trigger animations when the page loads
    setTimeout(() => {
        homeSection.classList.add('reveal');
    }, 100);

    // Particle animation (assuming you're using particles.js)
    particlesJS('home-particles', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#ffffff" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: "#ffffff", opacity: 0.4, width: 1 },
            move: { enable: true, speed: 2, direction: "none", random: false, straight: false, out_mode: "out", bounce: false }
        },
        interactivity: {
            detect_on: "canvas",
            events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true },
            modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
        },
        retina_detect: true
    });
});

// Add this to your existing script.js file

const sections = document.querySelectorAll('section');

const observerOptions = {
    root: null,
    threshold: 0.1,
    rootMargin: "-150px 0px -150px 0px"
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animate');
            setActiveNavItem(); // Update active nav item
        }
    });
}, observerOptions);

sections.forEach(section => {
    observer.observe(section);
});

// Add this to your existing script.js file

window.addEventListener('scroll', () => {
    const scrollProgress = document.getElementById('scroll-progress');
    const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
    const scrollHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrollPercentage = (scrollTop / scrollHeight) * 100;
    scrollProgress.style.width = scrollPercentage + '%';
});

function setActiveNavItem() {
    const sections = document.querySelectorAll('section');
    const navItems = document.querySelectorAll('.nav-links a');

    let currentSection = '';

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (window.pageYOffset >= sectionTop - 150) {
            currentSection = section.getAttribute('id');
        }
    });

    navItems.forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('href').substring(1) === currentSection) {
            item.classList.add('active');
        }
    });
}

// Call the function on scroll and on page load
window.addEventListener('scroll', setActiveNavItem);
window.addEventListener('load', setActiveNavItem);

// Add this to the end of your script.js file
window.addEventListener('load', () => {
    if (window.location.hash) {
        const targetSection = window.location.hash;
        setTimeout(() => {
            smoothScroll(targetSection, 1000);
        }, 100);
    }
});