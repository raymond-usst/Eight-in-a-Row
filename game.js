/* =========================================
   八子棋 (Eight-in-a-Row) - Game Engine
   ========================================= */

(() => {
    'use strict';

    // ---- Constants ----
    const BOARD_SIZE = 100;
    const WIN_LENGTH = 8;
    const CELL_SIZE = 32; // px per cell at zoom=1
    const PLAYERS = ['red', 'green', 'blue'];
    const PLAYER_LABELS = { red: '红方', green: '绿方', blue: '蓝方' };
    const PLAYER_COLORS = {
        red: { fill: '#e74c3c', stroke: '#c0392b', glow: 'rgba(231,76,60,0.5)' },
        green: { fill: '#27ae60', stroke: '#1e8449', glow: 'rgba(39,174,96,0.5)' },
        blue: { fill: '#2980b9', stroke: '#1f6694', glow: 'rgba(41,128,185,0.5)' },
    };

    // ---- Directions for line checks: [dr, dc] ----
    const DIRECTIONS = [
        [0, 1],   // horizontal
        [1, 0],   // vertical
        [1, 1],   // diagonal ↘
        [1, -1],  // diagonal ↙
    ];

    // ---- State ----
    let board = [];          // board[r][c] = null | 'red' | 'green' | 'blue'
    let currentPlayerIdx = 0;
    let moveHistory = [];
    let moveCount = 0;
    let gameOver = false;
    let winner = null;

    // AI Mode
    let gameMode = 'pvp';        // 'pvp' or 'ai'
    let humanColor = 'red';       // which color the human plays
    let aiConnected = false;
    let aiThinking = false;
    let aiWebSocket = null;
    const AI_SERVER_WS = 'ws://localhost:5000/api/ws/move';
    const AI_SERVER_HTTP = 'http://localhost:5000/api/status';
    const PLAYER_TO_ID = { red: 1, green: 2, blue: 3 };

    // Zoom & Pan
    let zoom = 1.0;
    const ZOOM_MIN = 0.15;
    const ZOOM_MAX = 3.0;
    let panX = 0, panY = 0;
    let isPanning = false;
    let panStartX, panStartY, panStartPanX, panStartPanY;

    // Hover
    let hoverRow = -1, hoverCol = -1;

    // Longest chain tracking
    let longestChains = { red: 0, green: 0, blue: 0 };
    let chainGroups = { red: {}, green: {}, blue: {} };  // { player: { length: count } }

    // Canvas & DOM refs
    let canvas, ctx;
    let minimapCanvas, minimapCtx;
    let boardPixelSize;

    // ---- Init ----
    function init() {
        canvas = document.getElementById('game-board');
        ctx = canvas.getContext('2d');
        minimapCanvas = document.getElementById('minimap');
        minimapCtx = minimapCanvas.getContext('2d');

        boardPixelSize = BOARD_SIZE * CELL_SIZE;
        canvas.width = boardPixelSize;
        canvas.height = boardPixelSize;
        minimapCanvas.width = 200;
        minimapCanvas.height = 200;

        resetGame();
        setupEvents();
        setupAIControls();
        centerBoard();
        render();
        renderMinimap();
        updateMinimapViewport();
    }

    function resetGame() {
        board = Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(null));
        currentPlayerIdx = 0;
        moveHistory = [];
        moveCount = 0;
        gameOver = false;
        winner = null;
        longestChains = { red: 0, green: 0, blue: 0 };
        chainGroups = { red: {}, green: {}, blue: {} };
        updateUI();
    }

    function centerBoard() {
        const viewport = document.getElementById('board-viewport');
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        // Start zoom so board fills most of the viewport
        zoom = Math.min(vw / boardPixelSize, vh / boardPixelSize) * 0.95;
        zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom));
        panX = (vw - boardPixelSize * zoom) / 2;
        panY = (vh - boardPixelSize * zoom) / 2;
        applyTransform();
    }

    // ---- Events ----
    function setupEvents() {
        const viewport = document.getElementById('board-viewport');

        // Mouse move (hover)
        viewport.addEventListener('mousemove', (e) => {
            const pos = screenToBoard(e.clientX, e.clientY);
            if (pos) {
                hoverRow = pos.row;
                hoverCol = pos.col;
                document.getElementById('hover-coords').textContent = `坐标: (${pos.col + 1}, ${pos.row + 1})`;
            } else {
                hoverRow = hoverCol = -1;
                document.getElementById('hover-coords').textContent = '坐标: -';
            }
            render();
        });

        // Click (place piece)
        viewport.addEventListener('click', (e) => {
            if (gameOver) return;
            if (isPanning) return;
            if (aiThinking) return;
            // In AI mode, only allow clicks on human's turn
            if (gameMode === 'ai' && PLAYERS[currentPlayerIdx] !== humanColor) return;
            const pos = screenToBoard(e.clientX, e.clientY);
            if (pos && board[pos.row][pos.col] === null) {
                commitMove(pos.row, pos.col, true);
            }
        });

        // Pan: mouse down / move / up
        viewport.addEventListener('mousedown', (e) => {
            if (e.button === 1 || e.button === 2 || (e.button === 0 && e.ctrlKey)) {
                isPanning = true;
                panStartX = e.clientX;
                panStartY = e.clientY;
                panStartPanX = panX;
                panStartPanY = panY;
                viewport.style.cursor = 'grabbing';
                e.preventDefault();
            }
        });
        window.addEventListener('mousemove', (e) => {
            if (isPanning) {
                panX = panStartPanX + (e.clientX - panStartX);
                panY = panStartPanY + (e.clientY - panStartY);
                applyTransform();
                updateMinimapViewport();
            }
        });
        window.addEventListener('mouseup', (e) => {
            if (isPanning) {
                isPanning = false;
                viewport.style.cursor = 'crosshair';
            }
        });

        // Touch support for panning
        let touchStartX, touchStartY, touchStartPanX2, touchStartPanY2;
        let isTouchPanning = false;
        let touchMoved = false;

        viewport.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                touchStartX = e.touches[0].clientX;
                touchStartY = e.touches[0].clientY;
                touchStartPanX2 = panX;
                touchStartPanY2 = panY;
                isTouchPanning = false;
                touchMoved = false;
            }
        }, { passive: true });

        viewport.addEventListener('touchmove', (e) => {
            if (e.touches.length === 1) {
                const dx = e.touches[0].clientX - touchStartX;
                const dy = e.touches[0].clientY - touchStartY;
                if (Math.abs(dx) > 5 || Math.abs(dy) > 5) {
                    isTouchPanning = true;
                    touchMoved = true;
                }
                if (isTouchPanning) {
                    panX = touchStartPanX2 + dx;
                    panY = touchStartPanY2 + dy;
                    applyTransform();
                    updateMinimapViewport();
                }
            }
            e.preventDefault();
        }, { passive: false });

        viewport.addEventListener('touchend', (e) => {
            if (!touchMoved && e.changedTouches.length === 1) {
                // Tap = place piece
                if (aiThinking) return;
                if (gameMode === 'ai' && PLAYERS[currentPlayerIdx] !== humanColor) return;
                const touch = e.changedTouches[0];
                const pos = screenToBoard(touch.clientX, touch.clientY);
                if (pos && !gameOver && (board[pos.row][pos.col] === null)) {
                    commitMove(pos.row, pos.col, true);
                }
            }
            isTouchPanning = false;
        });

        // Zoom (wheel)
        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = viewport.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;

            const oldZoom = zoom;
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, zoom * delta));

            // Zoom toward cursor
            panX = mx - (mx - panX) * (zoom / oldZoom);
            panY = my - (my - panY) * (zoom / oldZoom);

            applyTransform();
            render();
            renderMinimap();
            updateMinimapViewport();
        }, { passive: false });

        // Prevent context menu
        viewport.addEventListener('contextmenu', (e) => e.preventDefault());

        // Buttons
        document.getElementById('btn-undo').addEventListener('click', undoMove);
        document.getElementById('btn-restart').addEventListener('click', () => {
            resetGame();
            centerBoard();
            render();
            renderMinimap();
            updateMinimapViewport();
            document.getElementById('winner-modal').classList.add('hidden');
        });
        document.getElementById('btn-zoom-in').addEventListener('click', () => {
            zoom = Math.min(ZOOM_MAX, zoom * 1.3);
            applyTransform();
            render();
            renderMinimap();
            updateMinimapViewport();
        });
        document.getElementById('btn-zoom-out').addEventListener('click', () => {
            zoom = Math.max(ZOOM_MIN, zoom * 0.7);
            applyTransform();
            render();
            renderMinimap();
            updateMinimapViewport();
        });
        document.getElementById('btn-minimap').addEventListener('click', (e) => {
            const mc = document.getElementById('minimap-container');
            mc.classList.toggle('hidden');
            e.currentTarget.classList.toggle('active');
        });
        document.getElementById('btn-new-game').addEventListener('click', () => {
            resetGame();
            centerBoard();
            render();
            renderMinimap();
            updateMinimapViewport();
            document.getElementById('winner-modal').classList.add('hidden');
        });

        // Minimap click to navigate
        const minimapContainer = document.getElementById('minimap-container');
        minimapContainer.addEventListener('mousedown', (e) => {
            navigateFromMinimap(e);
        });
        minimapContainer.addEventListener('mousemove', (e) => {
            if (e.buttons === 1) navigateFromMinimap(e);
        });

        // Resize
        window.addEventListener('resize', () => {
            render();
            updateMinimapViewport();
        });
    }

    function navigateFromMinimap(e) {
        const rect = document.getElementById('minimap-container').getBoundingClientRect();
        const mx = (e.clientX - rect.left) / rect.width;
        const my = (e.clientY - rect.top) / rect.height;
        const viewport = document.getElementById('board-viewport');
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        panX = -(mx * boardPixelSize * zoom) + vw / 2;
        panY = -(my * boardPixelSize * zoom) + vh / 2;
        applyTransform();
        updateMinimapViewport();
        render();
    }

    function applyTransform() {
        canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
        canvas.style.transformOrigin = '0 0';
    }

    function screenToBoard(clientX, clientY) {
        const viewport = document.getElementById('board-viewport');
        const rect = viewport.getBoundingClientRect();
        const x = (clientX - rect.left - panX) / zoom;
        const y = (clientY - rect.top - panY) / zoom;
        const col = Math.floor(x / CELL_SIZE);
        const row = Math.floor(y / CELL_SIZE);
        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            return { row, col };
        }
        return null;
    }

    // ---- Game Logic ----
    function commitMove(row, col, isHuman = false) {
        if (gameOver) return;
        const player = PLAYERS[currentPlayerIdx];
        board[row][col] = player;
        moveHistory.push({ row, col, player });
        moveCount++;

        // Check win
        const maxChain = getMaxChainAt(row, col, player);
        recalcStats();

        if (maxChain >= WIN_LENGTH) {
            winner = player;
            gameOver = true;
            render();
            renderMinimap();
            showGameOverModal();
            return;
        }

        currentPlayerIdx = (currentPlayerIdx + 1) % 3;
        updateUI();
        render();
        renderMinimap();
        updateMinimapViewport();

        // Trigger AI turn if in AI mode. Also protect against infinite loop
        // by only automatically triggering AI turns if a human initiated it, or
        // if the recursive call detects it is an AI turn.
        if (isHuman && gameMode === 'ai' && !gameOver) {
            processAITurns();
        }
    }

    function undoMove() {
        if (moveHistory.length === 0 || gameOver || aiThinking) return;

        if (gameMode === 'ai') {
            do {
                const last = moveHistory.pop();
                board[last.row][last.col] = null;
                moveCount--;
                currentPlayerIdx = PLAYERS.indexOf(last.player);
            } while (moveHistory.length > 0 && PLAYERS[currentPlayerIdx] !== humanColor);
        } else {
            const last = moveHistory.pop();
            board[last.row][last.col] = null;
            moveCount--;
            currentPlayerIdx = PLAYERS.indexOf(last.player);
        }

        recalcStats();
        updateUI();
        render();
        renderMinimap();
    }

    function getMaxChainAt(row, col, player) {
        let maxLen = 1;
        for (const [dr, dc] of DIRECTIONS) {
            let count = 1;
            // Forward
            for (let i = 1; i < WIN_LENGTH; i++) {
                const nr = row + dr * i, nc = col + dc * i;
                if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
                if (board[nr][nc] !== player) break;
                count++;
            }
            // Backward
            for (let i = 1; i < WIN_LENGTH; i++) {
                const nr = row - dr * i, nc = col - dc * i;
                if (nr < 0 || nr >= BOARD_SIZE || nc < 0 || nc >= BOARD_SIZE) break;
                if (board[nr][nc] !== player) break;
                count++;
            }
            maxLen = Math.max(maxLen, count);
        }
        return maxLen;
    }

    /** Recalculate all chain stats for all players */
    function recalcStats() {
        longestChains = { red: 0, green: 0, blue: 0 };
        chainGroups = { red: {}, green: {}, blue: {} };

        // For each player, find all maximal chains
        for (const player of PLAYERS) {
            const visited = new Set();
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (board[r][c] !== player) continue;
                    for (const [dr, dc] of DIRECTIONS) {
                        const key = `${r},${c},${dr},${dc}`;
                        if (visited.has(key)) continue;
                        // Find start of chain in this direction
                        let sr = r, sc = c;
                        while (true) {
                            const pr = sr - dr, pc = sc - dc;
                            if (pr < 0 || pr >= BOARD_SIZE || pc < 0 || pc >= BOARD_SIZE) break;
                            if (board[pr][pc] !== player) break;
                            sr = pr; sc = pc;
                        }
                        // Count chain from start
                        let len = 0;
                        let cr = sr, cc = sc;
                        while (cr >= 0 && cr < BOARD_SIZE && cc >= 0 && cc < BOARD_SIZE && board[cr][cc] === player) {
                            visited.add(`${cr},${cc},${dr},${dc}`);
                            len++;
                            cr += dr;
                            cc += dc;
                        }
                        if (len >= 2) {
                            chainGroups[player][len] = (chainGroups[player][len] || 0) + 1;
                        }
                        longestChains[player] = Math.max(longestChains[player], len);
                    }
                }
            }
        }
    }

    /** Get ranking info for game-over modal */
    function getRanking() {
        const ranking = PLAYERS.map(p => ({
            player: p,
            label: PLAYER_LABELS[p],
            longest: longestChains[p],
            groups: { ...chainGroups[p] },
            isWinner: p === winner,
        }));

        ranking.sort((a, b) => {
            if (a.isWinner) return -1;
            if (b.isWinner) return 1;
            // Compare by longest chain
            if (b.longest !== a.longest) return b.longest - a.longest;
            // Collect all distinct chain lengths from both players, sorted descending
            const allLens = new Set([
                ...Object.keys(a.groups).map(Number),
                ...Object.keys(b.groups).map(Number),
            ]);
            const sorted = [...allLens].sort((x, y) => y - x);
            // Compare group counts at each length, from longest to shortest
            for (const len of sorted) {
                const aCount = a.groups[len] || 0;
                const bCount = b.groups[len] || 0;
                if (bCount !== aCount) return bCount - aCount;
            }
            return 0;
        });
        return ranking;
    }

    // ---- UI Updates ----
    function updateUI() {
        const current = PLAYERS[currentPlayerIdx];
        document.getElementById('turn-label').textContent = PLAYER_LABELS[current];
        document.getElementById('move-count').textContent = `第 ${moveCount + 1} 手`;
        const ti = document.getElementById('turn-indicator');
        const colors = PLAYER_COLORS[current];
        ti.style.background = `radial-gradient(circle at 35% 35%, ${colors.fill}, ${colors.stroke})`;

        // Update stats
        for (const p of PLAYERS) {
            document.getElementById(`stat-${p}`).textContent = longestChains[p];
        }

        // Active panel
        document.querySelectorAll('.player-panel').forEach(panel => {
            panel.classList.remove('active');
            if (panel.dataset.player === current) panel.classList.add('active');
        });
    }

    function showGameOverModal() {
        const ranking = getRanking();
        const medals = ['🥇', '🥈', '🥉'];
        const titles = ['冠军', '亚军', '季军'];

        document.getElementById('winner-title').textContent = `🏆 ${PLAYER_LABELS[winner]} 获胜!`;
        const list = document.getElementById('ranking-list');
        list.innerHTML = '';

        ranking.forEach((r, i) => {
            const item = document.createElement('div');
            item.className = 'rank-item';

            const colors = PLAYER_COLORS[r.player];
            let detailText = `最长连子: ${r.longest}`;
            // Add chain group info
            const sortedLens = Object.keys(r.groups).map(Number).sort((a, b) => b - a);
            if (sortedLens.length > 0) {
                const groupTexts = sortedLens.slice(0, 3).map(l => `${l}连×${r.groups[l]}`);
                detailText += ` | ${groupTexts.join(', ')}`;
            }

            item.innerHTML = `
                <span class="rank-medal">${medals[i]}</span>
                <div class="rank-color-dot" style="background: radial-gradient(circle at 35% 35%, ${colors.fill}, ${colors.stroke})"></div>
                <div class="rank-text">
                    <strong>${r.label} - ${titles[i]}</strong>
                    <div class="rank-detail">${detailText}</div>
                </div>
            `;
            list.appendChild(item);
        });

        document.getElementById('winner-modal').classList.remove('hidden');
    }

    // ---- Rendering ----
    let woodPattern = null;
    function createWoodPattern() {
        const patCanvas = document.createElement('canvas');
        patCanvas.width = 128;
        patCanvas.height = 128;
        const pctx = patCanvas.getContext('2d');

        // Base wood color
        pctx.fillStyle = '#d2a86e';
        pctx.fillRect(0, 0, 128, 128);

        // Wood grain lines
        for (let i = 0; i < 60; i++) {
            pctx.strokeStyle = `rgba(${139 + Math.random() * 40}, ${90 + Math.random() * 30}, ${43 + Math.random() * 20}, ${0.08 + Math.random() * 0.12})`;
            pctx.lineWidth = 0.5 + Math.random() * 2;
            pctx.beginPath();
            const y = Math.random() * 128;
            pctx.moveTo(0, y + (Math.random() - 0.5) * 10);
            for (let x = 0; x <= 128; x += 8) {
                pctx.lineTo(x, y + Math.sin(x * 0.05 + i) * (2 + Math.random() * 4) + (Math.random() - 0.5) * 2);
            }
            pctx.stroke();
        }

        // Subtle noise
        const imgData = pctx.getImageData(0, 0, 128, 128);
        for (let i = 0; i < imgData.data.length; i += 4) {
            const noise = (Math.random() - 0.5) * 12;
            imgData.data[i] += noise;
            imgData.data[i + 1] += noise;
            imgData.data[i + 2] += noise;
        }
        pctx.putImageData(imgData, 0, 0);

        woodPattern = ctx.createPattern(patCanvas, 'repeat');
    }

    function render() {
        if (!woodPattern) createWoodPattern();

        const viewport = document.getElementById('board-viewport');
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;

        // Calculate visible area in board coords
        const visLeft = Math.max(0, Math.floor(-panX / zoom / CELL_SIZE) - 1);
        const visTop = Math.max(0, Math.floor(-panY / zoom / CELL_SIZE) - 1);
        const visRight = Math.min(BOARD_SIZE, Math.ceil((vw - panX) / zoom / CELL_SIZE) + 1);
        const visBottom = Math.min(BOARD_SIZE, Math.ceil((vh - panY) / zoom / CELL_SIZE) + 1);

        // Clear visible area + margin
        const clearX = Math.max(0, visLeft * CELL_SIZE - CELL_SIZE);
        const clearY = Math.max(0, visTop * CELL_SIZE - CELL_SIZE);
        const clearW = (visRight - visLeft + 2) * CELL_SIZE;
        const clearH = (visBottom - visTop + 2) * CELL_SIZE;
        ctx.clearRect(clearX, clearY, clearW, clearH);

        // Wood background
        ctx.fillStyle = woodPattern;
        ctx.fillRect(clearX, clearY, clearW, clearH);

        // Grid lines
        ctx.strokeStyle = 'rgba(101, 67, 33, 0.35)';
        ctx.lineWidth = 0.5;

        for (let r = visTop; r <= visBottom && r < BOARD_SIZE; r++) {
            const y = r * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath();
            ctx.moveTo(visLeft * CELL_SIZE, y);
            ctx.lineTo(visRight * CELL_SIZE, y);
            ctx.stroke();
        }
        for (let c = visLeft; c <= visRight && c < BOARD_SIZE; c++) {
            const x = c * CELL_SIZE + CELL_SIZE / 2;
            ctx.beginPath();
            ctx.moveTo(x, visTop * CELL_SIZE);
            ctx.lineTo(x, visBottom * CELL_SIZE);
            ctx.stroke();
        }

        // Coordinate labels (every 10)
        if (zoom > 0.3) {
            ctx.fillStyle = 'rgba(101, 67, 33, 0.6)';
            ctx.font = `${Math.max(8, CELL_SIZE * 0.3)}px Inter, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            for (let c = visLeft; c < visRight; c++) {
                if ((c + 1) % 10 === 0) {
                    ctx.fillText(String(c + 1), c * CELL_SIZE + CELL_SIZE / 2, visTop * CELL_SIZE - 2);
                }
            }
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';
            for (let r = visTop; r < visBottom; r++) {
                if ((r + 1) % 10 === 0) {
                    ctx.fillText(String(r + 1), visLeft * CELL_SIZE - 4, r * CELL_SIZE + CELL_SIZE / 2);
                }
            }
        }

        // Star points (like Go board) at intersections every 20
        if (zoom > 0.25) {
            ctx.fillStyle = 'rgba(101, 67, 33, 0.5)';
            for (let r = 19; r < BOARD_SIZE; r += 20) {
                for (let c = 19; c < BOARD_SIZE; c += 20) {
                    if (r >= visTop && r < visBottom && c >= visLeft && c < visRight) {
                        ctx.beginPath();
                        ctx.arc(c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2, 3, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }
        }

        // Pieces
        const pieceRadius = CELL_SIZE * 0.4;
        for (let r = visTop; r < visBottom; r++) {
            for (let c = visLeft; c < visRight; c++) {
                const player = board[r][c];
                if (!player) continue;
                drawPiece(c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2, pieceRadius, player, false);
            }
        }

        // Hover indicator
        if (!gameOver && hoverRow >= 0 && hoverCol >= 0 && board[hoverRow][hoverCol] === null) {
            const current = PLAYERS[currentPlayerIdx];
            const cx = hoverCol * CELL_SIZE + CELL_SIZE / 2;
            const cy = hoverRow * CELL_SIZE + CELL_SIZE / 2;
            ctx.globalAlpha = 0.4;
            drawPiece(cx, cy, pieceRadius, current, true);
            ctx.globalAlpha = 1.0;
        }

        // Last move indicator
        if (moveHistory.length > 0) {
            const last = moveHistory[moveHistory.length - 1];
            const lx = last.col * CELL_SIZE + CELL_SIZE / 2;
            const ly = last.row * CELL_SIZE + CELL_SIZE / 2;
            ctx.strokeStyle = '#ffd700';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(lx, ly, pieceRadius + 3, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    function drawPiece(x, y, radius, player, isGhost) {
        const colors = PLAYER_COLORS[player];

        // Shadow
        if (!isGhost) {
            ctx.fillStyle = 'rgba(0,0,0,0.25)';
            ctx.beginPath();
            ctx.arc(x + 1.5, y + 2, radius, 0, Math.PI * 2);
            ctx.fill();
        }

        // Main fill with gradient for 3D effect
        const grad = ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.3, radius * 0.1, x, y, radius);
        grad.addColorStop(0, lightenColor(colors.fill, 40));
        grad.addColorStop(0.5, colors.fill);
        grad.addColorStop(1, colors.stroke);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Rim
        ctx.strokeStyle = colors.stroke;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Highlight
        if (!isGhost) {
            ctx.fillStyle = 'rgba(255,255,255,0.25)';
            ctx.beginPath();
            ctx.arc(x - radius * 0.2, y - radius * 0.25, radius * 0.35, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    function lightenColor(hex, amt) {
        let r = parseInt(hex.slice(1, 3), 16);
        let g = parseInt(hex.slice(3, 5), 16);
        let b = parseInt(hex.slice(5, 7), 16);
        r = Math.min(255, r + amt);
        g = Math.min(255, g + amt);
        b = Math.min(255, b + amt);
        return `rgb(${r},${g},${b})`;
    }

    // ---- Minimap ----
    function renderMinimap() {
        const w = minimapCanvas.width;
        const h = minimapCanvas.height;
        minimapCtx.clearRect(0, 0, w, h);

        // Background
        minimapCtx.fillStyle = '#c8a86e';
        minimapCtx.fillRect(0, 0, w, h);

        // Grid lines (sparse)
        minimapCtx.strokeStyle = 'rgba(101, 67, 33, 0.3)';
        minimapCtx.lineWidth = 0.5;
        for (let i = 0; i <= 10; i++) {
            const pos = (i / 10) * w;
            minimapCtx.beginPath();
            minimapCtx.moveTo(pos, 0);
            minimapCtx.lineTo(pos, h);
            minimapCtx.stroke();
            minimapCtx.beginPath();
            minimapCtx.moveTo(0, pos);
            minimapCtx.lineTo(w, pos);
            minimapCtx.stroke();
        }

        // Draw pieces
        const scale = w / BOARD_SIZE;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                const player = board[r][c];
                if (!player) continue;
                minimapCtx.fillStyle = PLAYER_COLORS[player].fill;
                minimapCtx.fillRect(c * scale, r * scale, Math.max(scale, 1.5), Math.max(scale, 1.5));
            }
        }
    }

    function updateMinimapViewport() {
        const viewport = document.getElementById('board-viewport');
        const vw = viewport.clientWidth;
        const vh = viewport.clientHeight;
        const minimapW = 200;
        const minimapH = 200;
        const totalW = boardPixelSize * zoom;
        const totalH = boardPixelSize * zoom;

        const left = (-panX / totalW) * minimapW;
        const top = (-panY / totalH) * minimapH;
        const width = (vw / totalW) * minimapW;
        const height = (vh / totalH) * minimapH;

        const rect = document.getElementById('minimap-viewport-rect');
        rect.style.left = Math.max(0, left) + 'px';
        rect.style.top = Math.max(0, top) + 'px';
        rect.style.width = Math.min(minimapW, width) + 'px';
        rect.style.height = Math.min(minimapH, height) + 'px';
    }

    // ---- AI Integration ----
    function setupAIControls() {
        // Mode buttons
        document.getElementById('btn-mode-pvp').addEventListener('click', () => {
            gameMode = 'pvp';
            document.getElementById('btn-mode-pvp').classList.add('active');
            document.getElementById('btn-mode-ai').classList.remove('active');
            document.getElementById('ai-options').classList.add('hidden');
            resetGame();
            centerBoard();
            render();
            renderMinimap();
            updateMinimapViewport();
        });

        document.getElementById('btn-mode-ai').addEventListener('click', () => {
            gameMode = 'ai';
            document.getElementById('btn-mode-ai').classList.add('active');
            document.getElementById('btn-mode-pvp').classList.remove('active');
            document.getElementById('ai-options').classList.remove('hidden');
            resetGame();
            centerBoard();
            render();
            renderMinimap();
            updateMinimapViewport();
            checkAIConnection();
            // If human is not first player, trigger AI
            if (PLAYERS[currentPlayerIdx] !== humanColor) {
                processAITurns();
            }
        });

        document.getElementById('human-color').addEventListener('change', (e) => {
            humanColor = e.target.value;
            resetGame();
            centerBoard();
            render();
            renderMinimap();
            updateMinimapViewport();
            if (gameMode === 'ai' && PLAYERS[currentPlayerIdx] !== humanColor) {
                processAITurns();
            }
        });

        document.getElementById('btn-connect-ai').addEventListener('click', checkAIConnection);
    }

    async function checkAIConnection() {
        const statusEl = document.getElementById('ai-status');
        try {
            const resp = await fetch(AI_SERVER_HTTP, { signal: AbortSignal.timeout(3000) });
            if (resp.ok) {
                const data = await resp.json();
                aiConnected = true;
                statusEl.textContent = `● AI 已连接 (${data.model_loaded ? '模型已加载' : '未加载模型'})`;
                statusEl.className = 'ai-status connected';
            } else {
                throw new Error('Server error');
            }
        } catch (e) {
            aiConnected = false;
            statusEl.textContent = '● AI 未连接';
            statusEl.className = 'ai-status disconnected';
        }
    }

    async function processAITurns() {
        // Prevent multiple simultaneous processAITurns loops
        if (window._aiProcessing) return;
        window._aiProcessing = true;

        try {
            // Process consecutive AI turns (since there are 2 AI players)
            while (!gameOver && gameMode === 'ai' && PLAYERS[currentPlayerIdx] !== humanColor) {
                await requestAIMove();
            }
        } finally {
            window._aiProcessing = false;
        }
    }

    async function requestAIMove() {
        if (gameOver || aiThinking) return;

        aiThinking = true;
        showThinking(true);

        // Detach old WebSocket handlers before closing to prevent stale
        // onclose from firing and placing phantom moves.
        if (aiWebSocket) {
            aiWebSocket.onopen = null;
            aiWebSocket.onmessage = null;
            aiWebSocket.onerror = null;
            aiWebSocket.onclose = null;
            aiWebSocket.close();
            aiWebSocket = null;
        }

        return new Promise((resolve) => {
            let resolved = false;
            const safeResolve = () => { if (!resolved) { resolved = true; resolve(); } };

            const ws = new WebSocket(AI_SERVER_WS);
            aiWebSocket = ws;

            ws.onopen = () => {
                // Guard: only act if this is still the active WebSocket
                if (ws !== aiWebSocket) return;
                const boardData = getBoardForAPI();
                const currentPid = PLAYER_TO_ID[PLAYERS[currentPlayerIdx]];
                ws.send(JSON.stringify({
                    board: boardData,
                    current_player: currentPid,
                    move_history: moveHistory.map(m => [m.row, m.col, PLAYER_TO_ID[m.player]]),
                }));
            };

            ws.onmessage = (event) => {
                if (ws !== aiWebSocket) return;
                const data = JSON.parse(event.data);
                if (data.type === 'progress') {
                    const el = document.getElementById('ai-thinking');
                    el.textContent = '🧠 ' + data.message;
                } else if (data.type === 'result') {
                    aiThinking = false;
                    showThinking(false);
                    if (data.row >= 0 && data.row < BOARD_SIZE &&
                        data.col >= 0 && data.col < BOARD_SIZE &&
                        board[data.row][data.col] === null) {
                        commitMove(data.row, data.col);
                    } else {
                        console.warn('AI returned invalid move, picking random');
                        placeRandomMove();
                    }
                    safeResolve();
                } else if (data.error) {
                    console.error('AI error:', data.error);
                    aiThinking = false;
                    showThinking(false);
                    placeRandomMove();
                    safeResolve();
                }
            };

            ws.onerror = (error) => {
                if (ws !== aiWebSocket) return;
                console.error('WebSocket Error:', error);
                aiThinking = false;
                showThinking(false);
                placeRandomMove();
                safeResolve();
            };

            ws.onclose = () => {
                if (ws !== aiWebSocket) return;
                if (aiThinking) {
                    aiThinking = false;
                    showThinking(false);
                    placeRandomMove();
                    safeResolve();
                }
            };
        });
    }

    function placeRandomMove() {
        // Find a legal move near the center of activity
        const center = getActiveCenter();
        const range = 15;
        for (let dist = 0; dist < range; dist++) {
            for (let dr = -dist; dr <= dist; dr++) {
                for (let dc = -dist; dc <= dist; dc++) {
                    if (Math.abs(dr) !== dist && Math.abs(dc) !== dist) continue;
                    const r = center.row + dr, c = center.col + dc;
                    if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === null) {
                        commitMove(r, c);
                        return;
                    }
                }
            }
        }
        // Absolute fallback
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] === null) {
                    commitMove(r, c);
                    return;
                }
            }
        }
    }

    function getActiveCenter() {
        if (moveHistory.length === 0) return { row: 50, col: 50 };
        let sumR = 0, sumC = 0;
        const recent = moveHistory.slice(-10);
        for (const m of recent) { sumR += m.row; sumC += m.col; }
        return { row: Math.round(sumR / recent.length), col: Math.round(sumC / recent.length) };
    }

    function getBoardForAPI() {
        // Convert board to 2D array of ints: 0=empty, 1=red, 2=green, 3=blue
        const data = [];
        for (let r = 0; r < BOARD_SIZE; r++) {
            const row = [];
            for (let c = 0; c < BOARD_SIZE; c++) {
                row.push(board[r][c] ? PLAYER_TO_ID[board[r][c]] : 0);
            }
            data.push(row);
        }
        return data;
    }

    function showThinking(show) {
        const el = document.getElementById('ai-thinking');
        if (show) {
            el.classList.remove('hidden');
        } else {
            el.classList.add('hidden');
        }
    }

    // ---- Start ----
    window.addEventListener('DOMContentLoaded', init);
})();
