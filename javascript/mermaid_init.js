// Mermaid initialization module
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';

mermaid.initialize({ 
    startOnLoad: false,
});

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initMermaid);
} else {
    initMermaid();
}

async function initMermaid() {
    // Find all mermaid code blocks
    const mermaidBlocks = document.querySelectorAll('pre > code.language-mermaid, .language-mermaid');
    
    mermaidBlocks.forEach((block, index) => {
        // Create a div for mermaid to render into
        const mermaidDiv = document.createElement('div');
        mermaidDiv.className = 'mermaid';
        mermaidDiv.textContent = block.textContent;
        
        // Replace the code block with the mermaid div
        if (block.parentElement.tagName === 'PRE') {
            block.parentElement.replaceWith(mermaidDiv);
        } else {
            block.replaceWith(mermaidDiv);
        }
    });
    
    // Run mermaid on all .mermaid divs
    await mermaid.run({
        querySelector: '.mermaid'
    });
}
