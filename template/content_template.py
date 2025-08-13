selectors = """
#slide{index}:checked ~ .content-area .slide:nth-child({index})
""".strip()

images = """
label[for="slide{index}"] {{ background-image: url('{path}'); }}
""".strip()

thumbnail_inp = """
<input type="radio" id="slide{index}" name="gallery" {status}>
""".strip()

thumbnail_nav = """
<label for="slide{index}"></label>
""".strip()

slides = """
<div class="slide">
    <div class="slide-image-container">
        <img src="{path}" class="slide-background-image">
        <img src="{path}" class="slide-fitted-image">
    </div>
    <div class="slide-content" alt="{title}">
        <div class="slide-description">{description}</div>
    </div>
</div>
""".strip()
