document.addEventListener('DOMContentLoaded', function() {
    // Handle video thumbnails
    const videoWrappers = document.querySelectorAll('.video-wrapper');
    
    videoWrappers.forEach(wrapper => {
        const thumbnail = wrapper.querySelector('.video-thumbnail');
        const playButton = wrapper.querySelector('.play-button');
        const videoFrame = wrapper.querySelector('.video-frame');
        
        if (thumbnail && playButton && videoFrame) {
            thumbnail.addEventListener('click', function() {
                // Hide thumbnail and play button
                thumbnail.style.display = 'none';
                playButton.style.display = 'none';
                
                // Show and play video
                videoFrame.style.display = 'block';
                videoFrame.src = videoFrame.src + '&autoplay=1';
            });
        }
    });

    // Handle image paths
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        if (img.src.includes('/img/2025/memento/')) {
            // Update image paths to use the data directory
            const newSrc = img.src.replace('/img/2025/memento/', 'data/');
            img.src = newSrc;
        }
    });

    // Get all navigation links
    const navLinks = document.querySelectorAll('.article-nav a');
    
    // Add click event listener to each link
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            // Remove active class from all links
            navLinks.forEach(l => l.classList.remove('active'));
            // Add active class to clicked link
            this.classList.add('active');
        });
    });

    // Handle scroll to update active state
    window.addEventListener('scroll', function() {
        const sections = document.querySelectorAll('section[id]');
        const scrollPosition = window.scrollY;

        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100; // Offset for better trigger point
            const sectionBottom = sectionTop + section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
                // Remove active class from all links
                navLinks.forEach(link => link.classList.remove('active'));
                // Add active class to corresponding link
                document.querySelector(`.article-nav a[href="#${sectionId}"]`).classList.add('active');
            }
        });
    });

    // Handle pipeline diagram interaction
    const pipelineSteps = document.querySelectorAll('.pipeline-step');
    const stepDetails = document.querySelectorAll('.step-details');
    const pipelineDetails = document.querySelector('.pipeline-details');

    // Set first step as active by default
    if (pipelineSteps.length > 0) {
        pipelineSteps[0].classList.add('active');
        stepDetails[0].classList.add('active');
        pipelineDetails.classList.add('active');
    }

    pipelineSteps.forEach(step => {
        step.addEventListener('click', function() {
            const stepNumber = this.getAttribute('data-step');
            
            // Update active state of steps
            pipelineSteps.forEach(s => s.classList.remove('active'));
            this.classList.add('active');
            
            // Update visible details
            stepDetails.forEach(detail => {
                detail.classList.remove('active');
                if (detail.getAttribute('data-step') === stepNumber) {
                    detail.classList.add('active');
                    pipelineDetails.classList.add('active');
                }
            });
        });
    });
});
