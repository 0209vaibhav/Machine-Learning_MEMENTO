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

// Moved from index.html

document.addEventListener('DOMContentLoaded', function() {
    // Helper to load and pretty-print JSON
    function loadJsonPreview(url, containerId) {
        fetch(url)
            .then(response => response.json())
            .then(data => {
                const container = document.getElementById(containerId);
                if (!container) return;
                container.innerHTML = `<pre><code>${JSON.stringify(data, null, 2)}</code></pre>`;
            })
            .catch(err => {
                const container = document.getElementById(containerId);
                if (container) container.innerHTML = '<em>Could not load metadata.</em>';
            });
    }

    // Helper to load and pretty-print CSV
    function loadCsvPreview(url, containerId) {
        fetch(url)
            .then(response => response.text())
            .then(csv => {
                const container = document.getElementById(containerId);
                if (!container) return;
                // Convert CSV to JSON (simple version)
                const lines = csv.split('\n');
                const headers = lines[0].split(',');
                const json = lines.slice(1).map(line => {
                    const values = line.split(',');
                    return headers.reduce((obj, header, index) => {
                        obj[header] = values[index];
                        return obj;
                    }, {});
                });
                container.innerHTML = `<pre><code>${JSON.stringify(json, null, 2)}</code></pre>`;
            })
            .catch(err => {
                const container = document.getElementById(containerId);
                if (container) container.innerHTML = '<em>Could not load metadata.</em>';
            });
    }

    // Load Step 1 outputs
    loadJsonPreview('ml_pipeline/input/step1_data_loader/memento_categories.json', 'categories-json-preview');
    loadJsonPreview('ml_pipeline/input/step1_data_loader/memento_tags.json', 'tags-json-preview');
    loadJsonPreview('ml_pipeline/input/step1_data_loader/memento_durations.json', 'durations-json-preview');
    loadCsvPreview('ml_pipeline/output/step1_data_processing/processed_data/user_mementos_processed.csv', 'outputs-json-preview');

    // Load Step 2 outputs
    loadJsonPreview('ml_pipeline/output/step2_model_training/models/model_info.json', 'step2-model-info-preview');
    loadJsonPreview('ml_pipeline/output/step2_model_training/metrics/model_metrics.json', 'step2-metrics-preview');

    // Load Step 3 outputs
    loadJsonPreview('ml_pipeline/output/step3_model_training/reports/model_report.json', 'step3-report-preview');

    // Load Step 4 outputs
    loadJsonPreview('ml_pipeline/output/step4_scraped_data/raw_data/scraped_mementos.json', 'step4-outputs-preview');
});

// Step 5 dynamic outputs
function loadStep5DynamicOutputs() {
    // CSV preview
    fetch('ml_pipeline/output/step5_processed_data/processed_data/processed_scraped_data.csv')
        .then(response => response.text())
        .then(csv => {
            const container = document.getElementById('step5-csv-preview');
            if (container) {
                // Show first 20 lines for preview
                const lines = csv.split('\n').slice(0, 20).join('\n');
                container.innerHTML = `<pre><code>${lines}</code></pre>`;
            }
        })
        .catch(() => {
            const container = document.getElementById('step5-csv-preview');
            if (container) container.innerHTML = '<em>Could not load CSV.</em>';
        });
    // JSON preview
    fetch('ml_pipeline/output/step5_processed_data/processed_data/processed_scraped_data.json')
        .then(response => response.json())
        .then(json => {
            const container = document.getElementById('step5-json-preview');
            if (container) container.innerHTML = `<pre><code>${JSON.stringify(json, null, 2).slice(0, 5000)}${json.length > 100 ? '\n... (truncated)' : ''}</code></pre>`;
        })
        .catch(() => {
            const container = document.getElementById('step5-json-preview');
            if (container) container.innerHTML = '<em>Could not load JSON.</em>';
        });
    // Report preview
    fetch('ml_pipeline/output/step5_processed_data/reports/processing_report.json')
        .then(response => response.json())
        .then(json => {
            const container = document.getElementById('step5-report-preview');
            if (container) container.innerHTML = `<pre><code>${JSON.stringify(json, null, 2)}</code></pre>`;
        })
        .catch(() => {
            const container = document.getElementById('step5-report-preview');
            if (container) container.innerHTML = '<em>Could not load report.</em>';
        });
}
document.addEventListener('DOMContentLoaded', loadStep5DynamicOutputs);

// Results Section dynamic dataset loading

document.addEventListener('DOMContentLoaded', function() {
    // Function to load and process dataset
    async function loadDataset(filePath, categoryId) {
        try {
            const response = await fetch(filePath);
            if (!response.ok) {
                const previewEl = document.getElementById(`${categoryId}-preview`);
                if (previewEl) previewEl.innerHTML = '<em>Dataset file not found.</em>';
                const countEl = document.getElementById(`${categoryId}-count`);
                if (countEl) countEl.textContent = '0';
                return [];
            }
            const data = await response.json();
            
            // Update count in heading
            const countEl = document.getElementById(`${categoryId}-count`);
            if (countEl) countEl.textContent = data.length;
            
            // Show entire dataset
            const previewEl = document.getElementById(`${categoryId}-preview`);
            if (previewEl) {
                previewEl.innerHTML = `<pre><code>${JSON.stringify(data, null, 2)}</code></pre>`;
            }
            
            return data;
        } catch (error) {
            console.error(`Error loading ${categoryId} dataset:`, error);
            const previewEl = document.getElementById(`${categoryId}-preview`);
            if (previewEl) previewEl.innerHTML = '<em>Error loading dataset</em>';
            const countEl = document.getElementById(`${categoryId}-count`);
            if (countEl) countEl.textContent = '0';
            return [];
        }
    }

    // Load all datasets
    Promise.all([
        loadDataset('ml_pipeline/Dataset/mementos_culture_all_pages.json', 'culture'),
        loadDataset('ml_pipeline/Dataset/mementos_escapes_all_pages.json', 'escapes'),
        loadDataset('ml_pipeline/Dataset/mementos_food_drink_all_pages.json', 'food-drink'),
        loadDataset('ml_pipeline/Dataset/mementos_things_to_do_all_pages.json', 'things-to-do'),
        loadDataset('ml_pipeline/Dataset/mementos_top_news_all_pages.json', 'top-news'),
        loadDataset('ml_pipeline/Dataset/mementos_wellness_nature_all_pages.json', 'wellness-nature')
    ]).then(datasets => {
    });
});
