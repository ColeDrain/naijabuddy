/* Interactive JS Logic for NaijaBuddy Conversational Recommender Dashboard */

// State variables
let activePersona = null;
let personas = [];
let currentDomainTab = 'Yelp';

// DOM Elements
const personaList = document.getElementById('personaList');
const activeUserName = document.getElementById('activeUserName');
const activeUserSub = document.getElementById('activeUserSub');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const alphaSlider = document.getElementById('alphaSlider');
const alphaVal = document.getElementById('alphaVal');
const statUserMean = document.getElementById('statUserMean');
const statCalType = document.getElementById('statCalType');
const catalogList = document.getElementById('catalogList');
const createCustomBtn = document.getElementById('createCustomBtn');
const customName = document.getElementById('customName');
const customPersona = document.getElementById('customPersona');

// 1. Initial Load & Setup
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
});

function initApp() {
    loadPersonas();
    loadCatalogTab(currentDomainTab);
}

function setupEventListeners() {
    // Chat Input Enter Key
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });

    // Chat Send Button
    sendBtn.addEventListener('click', handleSendMessage);

    // Alpha Slider Real-time Interaction
    alphaSlider.addEventListener('input', () => {
        const value = (alphaSlider.value / 100).toFixed(2);
        alphaVal.textContent = value;
        updateFormulaUI(value);
    });

    // Custom Persona Button
    createCustomBtn.addEventListener('click', handleCreateCustomPersona);
}

// 2. Load Personas from API
async function loadPersonas() {
    try {
        const response = await fetch('/api/users');
        if (!response.ok) throw new Error("Failed to load personas.");
        personas = await response.ok ? await response.json() : [];
        
        if (personas.length === 0) {
            personaList.innerHTML = `<div class="error-msg"><i class="fa-solid fa-circle-exclamation"></i> No personas seeded. Run data seeder first!</div>`;
            return;
        }

        renderPersonas();
        // Select first persona by default
        selectPersona(personas[0]);
    } catch (err) {
        console.error(err);
        personaList.innerHTML = `<div class="error-msg"><i class="fa-solid fa-circle-exclamation"></i> Error connecting to backend API.</div>`;
    }
}

function renderPersonas() {
    personaList.innerHTML = '';
    personas.forEach(p => {
        const card = document.createElement('div');
        card.className = `persona-card ${activePersona && activePersona.id === p.id ? 'active' : ''}`;
        card.innerHTML = `
            <h3>${escapeHTML(p.name)}</h3>
            <p>${escapeHTML(p.persona)}</p>
        `;
        card.addEventListener('click', () => selectPersona(p));
        personaList.appendChild(card);
    });
}

function selectPersona(persona) {
    activePersona = persona;
    renderPersonas();
    
    // Update active header
    activeUserName.textContent = persona.name;
    activeUserSub.textContent = "Active Simulation Persona";
    
    // Update Stats Panel
    statUserMean.textContent = persona.user_mean_rating ? persona.user_mean_rating.toFixed(2) : "N/A";
    statCalType.textContent = persona.id <= 3 ? "Warm Start" : "Cold Start";
    
    // Reset welcome message or customize
    appendBotMessage(`<strong>NaijaBuddy:</strong> You have selected <strong>${escapeHTML(persona.name)}</strong>. Ask me to recommend items, or type a custom query. Try clicking the shortcut buttons below!`);
    updateFormulaUI((alphaSlider.value / 100).toFixed(2));
}

// 3. Create Custom Cold-Start Persona
async function handleCreateCustomPersona() {
    const name = customName.value.trim();
    const desc = customPersona.value.trim();
    
    if (!name || !desc) {
        alert("Please provide both a name and taste description for the custom persona.");
        return;
    }
    
    const originalBtnText = createCustomBtn.innerHTML;
    createCustomBtn.disabled = true;
    createCustomBtn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Embedding tastes...`;
    
    try {
        const response = await fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, persona: desc })
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Failed to create persona");
        }
        
        const newPersona = await response.json();
        
        // Add to active persona array
        personas.push(newPersona);
        renderPersonas();
        selectPersona(newPersona);
        
        // Custom message reporting cold-start registration
        appendBotMessage(`<strong>NaijaBuddy:</strong> I have successfully registered custom persona <strong>${escapeHTML(newPersona.name)}</strong> in SQLite and computed its 384-dimensional vector embedding. <br>
        Since this user has zero historical rating records, any recommendation or review requests will automatically trigger our **Cold-Start Cluster Mean Calibration Layer**!`);
        
        // Reset inputs
        customName.value = '';
        customPersona.value = '';
        
        // Custom stat label
        statCalType.textContent = "Cold Start";
        statUserMean.textContent = "N/A (Using Cluster Mean)";
    } catch (err) {
        alert("Error: " + err.message);
    } finally {
        createCustomBtn.disabled = false;
        createCustomBtn.innerHTML = originalBtnText;
    }
}

// 4. Load Catalog Panel
async function loadCatalogTab(domain) {
    currentDomainTab = domain;
    
    // Update Tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.textContent.trim() === domain) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    
    catalogList.innerHTML = `<div class="loading-spinner"><i class="fa-solid fa-circle-notch fa-spin"></i> Loading...</div>`;
    
    try {
        const response = await fetch(`/api/items?domain=${domain}`);
        if (!response.ok) throw new Error();
        const items = await response.json();
        
        catalogList.innerHTML = '';
        items.forEach(item => {
            const row = document.createElement('div');
            row.className = 'catalog-row';
            row.innerHTML = `
                <span><strong>${escapeHTML(item.name)}</strong></span>
                <span>⭐ ${item.average_rating.toFixed(1)}</span>
            `;
            catalogList.appendChild(row);
        });
    } catch (err) {
        catalogList.innerHTML = `<div class="error-msg">Error loading catalog.</div>`;
    }
}

// 5. Conversational UI Helpers
function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg bubble-user';
    msg.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-user-astronaut"></i></div>
        <div class="msg-content">
            <p>${escapeHTML(text)}</p>
        </div>
    `;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function appendBotMessage(htmlContent) {
    const msg = document.createElement('div');
    msg.className = 'msg bubble-bot';
    msg.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content">
            <p>${htmlContent}</p>
        </div>
    `;
    chatMessages.appendChild(msg);
    scrollToBottom();
}

function appendTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'msg bubble-bot typing-indicator-msg';
    indicator.id = 'typingIndicator';
    indicator.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content" style="padding: 10px 16px;">
            <div class="loading-spinner" style="padding: 0; text-align: left; color: var(--text-muted);">
                <i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing tastes & loading recommendations...
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 6. Handle Chat Commands
function sendPrompt(text) {
    chatInput.value = text;
    handleSendMessage();
}

async function handleSendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    appendUserMessage(text);
    chatInput.value = '';
    
    // Command Matching
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('recommend spots on yelp') || (lowerText.includes('recommend') && lowerText.includes('yelp'))) {
        fetchRecommendations('Yelp');
    } else if (lowerText.includes('recommend movies on amazon') || (lowerText.includes('recommend') && lowerText.includes('amazon'))) {
        fetchRecommendations('Amazon');
    } else if (lowerText.includes('recommend books on goodreads') || (lowerText.includes('recommend') && lowerText.includes('goodreads'))) {
        fetchRecommendations('Goodreads');
    } else if (lowerText.startsWith('simulate review for ')) {
        // Advanced: Simulate review command
        const itemName = text.replace(/simulate review for /i, '').trim();
        handleSimulateReviewByName(itemName);
    } else {
        // Conversational response
        appendBotMessage(`<strong>NaijaBuddy:</strong> I hear you! To get personalized recommendations tailored to your active persona, please choose one of the quick actions below, or ask me: <br>
        • <em>"Recommend spots on Yelp"</em> <br>
        • <em>"Recommend movies on Amazon"</em> <br>
        • <em>"Recommend books on Goodreads"</em>`);
    }
}

// 7. Recommender Action (Task B API Call)
async function fetchRecommendations(domain) {
    if (!activePersona) {
        appendBotMessage("<strong>NaijaBuddy:</strong> Abeg, select a persona on the left side first before running recommendations!");
        return;
    }
    
    appendTypingIndicator();
    
    try {
        const response = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_name: activePersona.name,
                domain: domain
            })
        });
        
        removeTypingIndicator();
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error");
        }
        
        const recommendations = await response.json();
        renderRecommendationCards(recommendations, domain);
    } catch (err) {
        removeTypingIndicator();
        appendBotMessage(`<strong>NaijaBuddy:</strong> E pain me, error fetching recommendations: <em>${escapeHTML(err.message)}</em>. Make sure GGUF model or fallback is active.`);
    }
}

function renderRecommendationCards(recommendations, domain) {
    const cardGrid = document.createElement('div');
    cardGrid.className = 'recommend-card-grid';
    
    recommendations.forEach(rec => {
        const card = document.createElement('div');
        card.className = 'recommend-card';
        if (rec.critic_penalty) {
            card.style.opacity = '0.7';
            card.style.borderColor = 'rgba(235, 94, 40, 0.4)';
        }
        
        const similarityPercentage = (rec.similarity * 100).toFixed(1);
        
        card.innerHTML = `
            <div class="card-rank">#${rec.rank}</div>
            <div class="card-body">
                <div class="card-header-row">
                    <h4 class="card-title">${escapeHTML(rec.name)}</h4>
                    <span class="card-category">${escapeHTML(rec.category)}</span>
                </div>
                <p class="card-desc">${escapeHTML(rec.description)}</p>
                <div class="card-justification" style="${rec.critic_penalty ? 'background: rgba(235,94,40,0.05); border-left-color: #eb5e28;' : ''}">
                    <i class="fa-solid ${rec.critic_penalty ? 'fa-triangle-exclamation' : 'fa-quote-left'}"></i>
                    <p class="justification-text">${escapeHTML(rec.explanation)}</p>
                </div>
                <div class="card-actions">
                    <span class="card-similarity"><i class="fa-solid fa-bolt"></i> Sim: ${similarityPercentage}%</span>
                    <button class="simulate-review-btn" onclick="executeReviewSimulation(${rec.id}, '${escapeHTML(rec.name)}')">
                        <i class="fa-solid fa-comments"></i> Simulate Review
                    </button>
                </div>
            </div>
        `;
        cardGrid.appendChild(card);
    });
    
    const container = document.createElement('div');
    container.className = 'msg bubble-bot';
    container.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content" style="width: 100%; max-width: 100%;">
            <p><strong>NaijaBuddy:</strong> Here are your Top recommendations for <strong>${domain}</strong>, curated specifically for <strong>${activePersona.name}</strong>:</p>
            <div id="cardsWrapper"></div>
        </div>
    `;
    
    chatMessages.appendChild(container);
    document.getElementById('cardsWrapper').id = ''; // clear temp ID
    container.querySelector('#cardsWrapper').appendChild(cardGrid);
    scrollToBottom();
}

// 8. Review Simulation Action (Task A API Call)
async function executeReviewSimulation(itemId, itemName) {
    if (!activePersona) return;
    
    // Add temporary message
    appendBotMessage(`<strong>NaijaBuddy:</strong> Simulating local rating & written review for <strong>${itemName}</strong>...`);
    appendTypingIndicator();
    
    const alpha = parseFloat(alphaSlider.value) / 100;
    
    try {
        const response = await fetch('/api/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_name: activePersona.name,
                item_id: itemId,
                alpha: alpha
            })
        });
        
        removeTypingIndicator();
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error");
        }
        
        const res = await response.json();
        renderSimulatedReviewCard(res, alpha);
    } catch (err) {
        removeTypingIndicator();
        appendBotMessage(`<strong>NaijaBuddy:</strong> Omo, review simulation failed: <em>${escapeHTML(err.message)}</em>`);
    }
}

async function handleSimulateReviewByName(itemName) {
    // Look up item in API
    try {
        const res = await fetch('/api/items');
        const items = await res.json();
        const found = items.find(it => it.name.toLowerCase() === itemName.toLowerCase() || it.name.toLowerCase().includes(itemName.toLowerCase()));
        
        if (found) {
            executeReviewSimulation(found.id, found.name);
        } else {
            appendBotMessage(`<strong>NaijaBuddy:</strong> I searched my catalog but couldn't find any item named <strong>"${escapeHTML(itemName)}"</strong>.`);
        }
    } catch (e) {
        appendBotMessage(`<strong>NaijaBuddy:</strong> Could not check catalog: <em>${e.message}</em>`);
    }
}

function renderSimulatedReviewCard(res, alpha) {
    // Generate star HTML
    const getStars = (rating) => {
        let stars = '';
        const rounded = Math.round(rating * 2) / 2;
        for (let i = 1; i <= 5; i++) {
            if (i <= rounded) {
                stars += '<i class="fa-solid fa-star" style="color: var(--accent-gold);"></i>';
            } else if (i - 0.5 === rounded) {
                stars += '<i class="fa-solid fa-star-half-stroke" style="color: var(--accent-gold);"></i>';
            } else {
                stars += '<i class="fa-regular fa-star" style="color: rgba(255,255,255,0.2);"></i>';
            }
        }
        return stars;
    };
    
    // Check baseline text based on Warm vs Cold
    let baselineText = '';
    let baselineVal = 3.5;
    
    if (activePersona.user_mean_rating) {
        baselineVal = activePersona.user_mean_rating;
        baselineText = `User Mean: <strong>${baselineVal.toFixed(2)}</strong> (Warm Start)`;
    } else {
        // Cold start, cluster mean fallback (using mock or real values)
        // If the response returns some details or we calculate locally
        baselineVal = 3.5; // default cluster mean
        baselineText = `Cluster Mean: <strong>${baselineVal.toFixed(2)}</strong> (Cold Start Fallback)`;
    }
    
    const rawRating = res.raw_rating;
    const finalCalibrated = res.calibrated_rating;
    
    const cardHtml = `
        <div class="simulation-result-card" style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-glass); border-radius: 16px; padding: 18px; margin: 12px 0; display: flex; flex-direction: column; gap: 14px; animation: fadeIn 0.4s ease-out;">
            <!-- Formula Math breakdown -->
            <div style="display: flex; flex-direction: column; gap: 4px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-muted);">
                    <span>Raw LLM Rating:</span>
                    <span>${rawRating.toFixed(2)} ${getStars(rawRating)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--text-muted);">
                    <span>${baselineText}:</span>
                    <span>${baselineVal.toFixed(2)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; color: var(--accent-primary); font-weight: 600; margin-top: 4px;">
                    <span>Calibrated Final Score:</span>
                    <span>${finalCalibrated.toFixed(2)} ${getStars(finalCalibrated)}</span>
                </div>
            </div>
            
            <!-- Mathematical blending explanation -->
            <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 8px; font-family: monospace; font-size: 10.5px; color: var(--accent-gold); text-align: center;">
                Final = (${alpha} × ${rawRating.toFixed(2)}) + (${(1-alpha).toFixed(2)} × ${baselineVal.toFixed(2)}) = <strong>${finalCalibrated.toFixed(2)}</strong>
            </div>
            
            <!-- Localized written review -->
            <div style="display: flex; gap: 10px; margin-top: 4px;">
                <div style="font-size: 20px; color: var(--accent-primary); align-self: flex-start;"><i class="fa-solid fa-quote-left"></i></div>
                <p style="font-size: 13px; font-style: italic; line-height: 1.5; color: var(--text-primary);">${escapeHTML(res.review)}</p>
            </div>
            
            <div style="font-size: 10px; text-align: right; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">
                Generated by: <strong>${escapeHTML(res.user_name)}</strong>
            </div>
        </div>
    `;
    
    appendBotMessage(cardHtml);
}

// 9. Dial Formula Updates
function updateFormulaUI(alpha) {
    const baseline = activePersona && activePersona.user_mean_rating ? activePersona.user_mean_rating.toFixed(2) : "3.50";
    const expr = document.querySelector('.math-expr');
    if (expr) {
        expr.innerHTML = `Final = (${alpha} × LLM) + (${(1 - alpha).toFixed(2)} × ${baseline})`;
    }
}

// 10. Utility Functions
function escapeHTML(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}
