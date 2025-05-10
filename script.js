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
});
