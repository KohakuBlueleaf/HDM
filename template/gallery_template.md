

<style>
.gallery-container {
    position: relative;
    width: 100%;
    margin: 0 auto;
    font-family: Arial, sans-serif;
}
.gallery-container img{
    margin: 0 !important;
    padding: 0 !important;
}

/* Hide radio buttons */
.gallery-container input[type="radio"] {
    display: none;
}

/* Thumbnail navigation */
.thumbnail-nav {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding: 10px;
    overflow-x: auto;
    scroll-behavior: smooth;
    scroll-snap-type: x mandatory;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    scrollbar-color: #B398F5 #f0f0f0;
}

/* Custom scrollbar for webkit browsers */
.thumbnail-nav::-webkit-scrollbar {
    height: 6px;
}

.thumbnail-nav::-webkit-scrollbar-track {
    background: #f0f0f0;
    border-radius: 3px;
}

.thumbnail-nav::-webkit-scrollbar-thumb {
    background: #B398F5;
    border-radius: 3px;
}

.thumbnail-nav::-webkit-scrollbar-thumb:hover {
    background: #4C88F5;
}

.thumbnail-nav label {
    width: 100px;
    height: 120px;
    border-radius: 8px;
    background-color: #B398F5;
    background-size: cover;
    background-position: center;
    cursor: pointer;
    transition: all 0.3s ease;
    display: block;
    flex-shrink: 0;
    scroll-snap-align: center;
    position: relative;
    overflow: hidden;
    padding: 0;
}

/* Blurred background for thumbnails */
.thumbnail-nav label::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: inherit;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    filter: blur(2px) brightness(25%);
    transform: scale(1.25);
    z-index: 1;
}

/* Fitted image for thumbnails */
.thumbnail-nav label::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100%;
    height: 100%;
    background-image: inherit;
    background-size: contain;
    background-position: center;
    background-repeat: no-repeat;
    z-index: 2;
}

.thumbnail-nav label:hover {
    border-color: #4C88F5;
    transform: scale(1.05);
}

.gallery-container input[type="radio"]:checked + label {
    border-color: #6296F5;
    border-width: 4px;
    transform: scale(1.1);
}

/* Main content area */
.content-area {
    position: relative;
    width: 100%;
    min-height: 780px;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

/* Individual slide containers */
.slide {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.5s ease;
    background: white;
}

.slide.active {
    opacity: 1;
    visibility: visible;
}

.slide-background-image {
    width: 100%;
    height: 480px;
    object-fit: cover;
    border-radius: 10px 10px 0 0;
    filter: blur(3px) brightness(25%);
    transform: scale(1.1);
    position: absolute;
    top: 0;
    left: 0;
    z-index: 1;
    margin: 0 !important;
    padding: 0 !important;
}

/* Container for the image area */
.slide-image-container {
    position: relative;
    width: 100%;
    height: 480px;
    overflow: hidden;
    border-radius: 10px 10px 0 0;
}

/* Fitted image overlay */
.slide-fitted-image {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    z-index: 5;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    margin: 0 !important;
    padding: 0 !important;
}

.slide-content {
    height: 300px;
    padding: 20px;
    flex-grow: 1;
    background-color: #21203b;
    overflow-y: scroll;
}

.slide-title {
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 10px;
    color: #eee;
}

.slide-description {
    color: #aaa;
    /* line-height: 1.6; */
}


/* CSS selectors to show active slides */
{selectors} {
    opacity: 1;
    visibility: visible;
}

/* Thumbnail background images */
{images}

/* Responsive design */
@media (max-width: 600px) {
    .thumbnail-nav {
        gap: 8px;
        padding: 8px;
    }
    
    .thumbnail-nav label {
        width: 80px;
        height: 60px;
    }
    
    .slide img {
        height: 200px;
    }
    
    .slide-content {
        padding: 15px;
    }
    
    .slide-title {
        font-size: 1.2em;
    }
}
</style>


<div class="gallery-container">
{thumbnail_inp}

<!-- Thumbnail navigation -->
<div class="thumbnail-nav">
    {thumbnail_nav}
</div>

<!-- Main content area with slides -->
<div class="content-area">
{slides}
</div>

</div>