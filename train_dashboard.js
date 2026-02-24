/**
 * Training Dashboard - Real-time visualization
 * Connects via WebSocket to MuZero training process
 *
 * Supports session-based scoring (3:0:-2 placement system)
 * and Best-of-5 round tracking with per-game/per-round view toggle.
 */
(function () {
    'use strict';

    // ---- Config ----
    const BOARD_SIZE = 100;
    const PLAYER_COLORS = { 1: '#e74c3c', 2: '#27ae60', 3: '#2980b9' };
    const PLAYER_NAMES = { 1: '红方', 2: '绿方', 3: '蓝方' };
    const PLAYER_GLOWS = { 1: '#ff6b6b', 2: '#6bff6b', 3: '#6bb5ff' };
    const PLACEMENT_POINTS = [5, 2, 0]; // 1st, 2nd, 3rd

    // ---- ELO Config ----
    const ELO_INITIAL = 1500;
    const ELO_K = 16;  // K-factor per pairwise matchup (each player faces 2 opponents per game)

    // ---- State ----
    let ws = null;
    let board = [];          // 100x100
    let boardCanvas, boardCtx;
    let gameStep = 0;
    let currentPlayer = 1;
    let totalGames = 0;
    let winCounts = { 1: 0, 2: 0, 3: 0, draw: 0 };
    let lossChart, statsChart, scoresChart, eloChart;
    let startTime = Date.now();
    let lastMove = null;

    // Session / Placement tracking (per-game)
    let placementCounts = {
        1: [0, 0, 0], // [1st, 2nd, 3rd] for player 1
        2: [0, 0, 0],
        3: [0, 0, 0],
    };
    let cumulativePoints = { 1: 0, 2: 0, 3: 0 };
    let rankedGames = 0; // games with ranking data
    let currentSessionInfo = null; // latest session context

    // Round-level tracking (per-round / Best-of-5)
    let roundPlacementCounts = {
        1: [0, 0, 0], // [Champion, Runner-up, Third] per round
        2: [0, 0, 0],
        3: [0, 0, 0],
    };
    let roundWinCounts = { 1: 0, 2: 0, 3: 0, draw: 0 };
    let totalRounds = 0;
    let lastSeenTotalRounds = 0; // detect round transitions from server data

    // View mode: 'game' or 'round'
    let winrateViewMode = 'game';
    let placementViewMode = 'game';

    // ELO tracking
    let eloRatings = { 1: ELO_INITIAL, 2: ELO_INITIAL, 3: ELO_INITIAL };
    let lastEloRankedGames = 0; // prevent re-computation with same data
    const eloHistory = { labels: [], red: [], green: [], blue: [] };
    const MAX_ELO_POINTS = 500;

    // Rolling score history for chart (game-by-game)
    const scoreHistory = { labels: [], red: [], green: [], blue: [] };
    const MAX_SCORE_POINTS = 300;
    // Per-round points (resets to 0 each round, driven by server data)
    let roundPoints = { 1: 0, 2: 0, 3: 0 };
    let roundGameIndex = 0; // game index within current round (0-4 for BO5)
    // All-time baselines at start of current round
    let roundBaselinePoints = { 1: 0, 2: 0, 3: 0 };
    let roundBaselineRankedGames = 0;

    // ---- Toggle Handler (exposed globally for onclick) ----
    window.__dashToggle = function (panel, mode) {
        if (panel === 'winrate') {
            winrateViewMode = mode;
            document.getElementById('wr-mode-game').classList.toggle('active', mode === 'game');
            document.getElementById('wr-mode-round').classList.toggle('active', mode === 'round');
            updateWinRates();
        } else if (panel === 'placement') {
            placementViewMode = mode;
            document.getElementById('plc-mode-game').classList.toggle('active', mode === 'game');
            document.getElementById('plc-mode-round').classList.toggle('active', mode === 'round');
            updatePlacementDisplay();
        }
    };

    // ---- Init ----
    function init() {
        boardCanvas = document.getElementById('live-board');
        boardCtx = boardCanvas.getContext('2d');
        resetBoard();
        initCharts();
        setupEvents();
        renderBoard();
    }

    function resetBoard() {
        board = Array.from({ length: BOARD_SIZE }, () => new Array(BOARD_SIZE).fill(0));
        gameStep = 0;
        lastMove = null;
    }

    // ---- WebSocket ----
    function setupEvents() {
        document.getElementById('btn-connect').addEventListener('click', connect);
        // Auto-connect on load
        setTimeout(connect, 500);
    }

    function connect() {
        const url = document.getElementById('ws-url').value;
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            ws.close();
        }

        setStatus('connecting');
        try {
            ws = new WebSocket(url);
        } catch (e) {
            setStatus('disconnected');
            return;
        }

        ws.onopen = () => setStatus('connected');
        ws.onclose = () => {
            setStatus('disconnected');
            // Auto-reconnect after 3s
            setTimeout(() => {
                if (!ws || ws.readyState === WebSocket.CLOSED) connect();
            }, 3000);
        };
        ws.onerror = () => setStatus('disconnected');
        ws.onmessage = (evt) => {
            try {
                const msg = JSON.parse(evt.data);
                handleMessage(msg);
            } catch (e) {
                console.warn('Bad WS message:', e);
            }
        };
    }

    function setStatus(status) {
        const dot = document.getElementById('ws-status');
        const label = document.getElementById('ws-label');
        const hint = document.getElementById('connection-hint');
        dot.className = 'status-dot ' + status;
        const labels = { connected: '已连接', disconnected: '未连接', connecting: '连接中...' };
        label.textContent = labels[status] || status;
        if (hint) hint.classList.toggle('hidden', status === 'connected');
    }

    // ---- Message Handler ----
    function handleMessage(msg) {
        switch (msg.type) {
            case 'selfplay_start':
                resetBoard();
                document.getElementById('board-info').textContent = `Game #${msg.game_index + 1}`;
                document.getElementById('game-iter').textContent = `Iteration: ${msg.iteration || '-'}`;
                renderBoard();
                break;

            case 'selfplay_move':
                placeStone(msg.row, msg.col, msg.player);
                lastMove = { row: msg.row, col: msg.col };
                gameStep++;
                currentPlayer = msg.player;
                document.getElementById('game-step').textContent = `步数: ${gameStep}`;
                document.getElementById('game-player').textContent = `落子: ${PLAYER_NAMES[msg.player] || '-'}`;
                renderBoard();
                break;

            case 'selfplay_end':
                totalGames++;
                if (msg.winner && msg.winner >= 1 && msg.winner <= 3) {
                    winCounts[msg.winner]++;
                } else {
                    winCounts.draw++;
                }
                updateWinRates();

                // Process rankings & session data
                if (msg.rankings && msg.rankings.length > 0) {
                    processRankings(msg.rankings);
                }
                if (msg.session) {
                    currentSessionInfo = msg.session;
                    updateSessionDisplay();
                }

                document.getElementById('board-info').textContent =
                    msg.winner ? `${PLAYER_NAMES[msg.winner]} 获胜 (${gameStep}步)` : `平局 (${gameStep}步)`;
                break;

            case 'round_end':
                // Best-of-5 round completed
                totalRounds++;
                if (msg.round_winner && msg.round_winner >= 1 && msg.round_winner <= 3) {
                    roundWinCounts[msg.round_winner]++;
                } else {
                    roundWinCounts.draw++;
                }
                // Resolve round_rankings: defensive parse (array of [pid, placement]) or fallback from round_winner/round_scores
                let resolvedRankings = [];
                if (msg.round_rankings && msg.round_rankings.length > 0) {
                    for (const entry of msg.round_rankings) {
                        const rawPid = Array.isArray(entry) ? entry[0] : (entry.pid != null ? entry.pid : entry[0]);
                        const rawPl = Array.isArray(entry) ? entry[1] : (entry.placement != null ? entry.placement : entry[1]);
                        const pid = Number(rawPid);
                        const placement = Number(rawPl);
                        if (!Number.isNaN(pid) && !Number.isNaN(placement) && pid >= 1 && pid <= 3 && placement >= 0 && placement <= 2)
                            resolvedRankings.push([pid, placement]);
                    }
                }
                if (resolvedRankings.length === 0 && (msg.round_winner || msg.round_scores)) {
                    const scores = msg.round_scores || { 1: 0, 2: 0, 3: 0 };
                    const winner = msg.round_winner;
                    if (winner >= 1 && winner <= 3) {
                        resolvedRankings.push([winner, 0]);
                        const others = [1, 2, 3].filter(p => p !== winner).sort((a, b) => (scores[b] || 0) - (scores[a] || 0));
                        others.forEach((pid, i) => resolvedRankings.push([pid, i + 1]));
                    } else {
                        const order = [1, 2, 3].sort((a, b) => (scores[b] || 0) - (scores[a] || 0));
                        order.forEach((pid, i) => resolvedRankings.push([pid, i]));
                    }
                }
                for (const [pid, placement] of resolvedRankings) {
                    if (pid >= 1 && pid <= 3 && placement >= 0 && placement <= 2)
                        roundPlacementCounts[pid][placement]++;
                }
                if (winrateViewMode === 'round') updateWinRates();
                if (placementViewMode === 'round') updatePlacementDisplay();
                // Score chart reset + ELO update now handled by onRoundsChanged via server data
                if (resolvedRankings.length >= 2)
                    updateEloFromRoundRankings(resolvedRankings);
                break;

            case 'training_metrics':
                updateLossChart(msg);
                updateStatsChart(msg);
                updateSummaryBar(msg);
                // Sync games/win counts from server so 累计胜率 matches 名次分布 (same source)
                if (msg.total_games !== undefined) totalGames = msg.total_games;
                if (msg.win_counts) winCounts = { ...winCounts, ...msg.win_counts };
                updateWinRates();
                // Sync authoritative round data from all actors
                if (msg.total_rounds !== undefined) {
                    totalRounds = msg.total_rounds;
                }
                if (msg.round_wins) {
                    roundWinCounts = {
                        1: msg.round_wins['1'] || 0,
                        2: msg.round_wins['2'] || 0,
                        3: msg.round_wins['3'] || 0,
                        draw: msg.round_wins['draw'] || 0,
                    };
                }
                if (msg.round_placements) {
                    for (const pid of [1, 2, 3]) {
                        const arr = msg.round_placements[String(pid)] || [0, 0, 0];
                        roundPlacementCounts[pid] = [arr[0] || 0, arr[1] || 0, arr[2] || 0];
                    }
                }
                // Detect round transition → update baselines + ELO (BEFORE score chart sync)
                if (msg.total_rounds !== undefined && msg.total_rounds > lastSeenTotalRounds) {
                    onRoundsChanged(msg.total_rounds - lastSeenTotalRounds);
                    lastSeenTotalRounds = msg.total_rounds;
                }
                // Sync authoritative placement data AFTER baseline update
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                }
                // Always update both views (game + round) so they stay fresh
                updateWinRates();
                updatePlacementDisplay();
                // Sync server-pushed league Elo (authoritative, single value)
                if (msg.league_elo !== undefined) {
                    const sumEl = document.getElementById('sum-elo');
                    if (sumEl) sumEl.textContent = `ELO: ${Math.round(msg.league_elo)}`;
                }
                break;

            case 'batch_stats':
                // Async mode: stats from all actors
                if (msg.total_games) totalGames = msg.total_games;
                if (msg.win_counts) winCounts = { ...winCounts, ...msg.win_counts };
                updateWinRates();
                // Sync placement/ELO from all actors
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                }
                break;

            case 'status':
                if (msg.total_games !== undefined) totalGames = msg.total_games;
                if (msg.win_counts) winCounts = { ...winCounts, ...msg.win_counts };
                updateSummaryBar(msg);
                // Sync authoritative round data
                if (msg.total_rounds !== undefined) totalRounds = msg.total_rounds;
                if (msg.round_wins) {
                    roundWinCounts = {
                        1: msg.round_wins['1'] || 0,
                        2: msg.round_wins['2'] || 0,
                        3: msg.round_wins['3'] || 0,
                        draw: msg.round_wins['draw'] || 0,
                    };
                }
                if (msg.round_placements) {
                    for (const pid of [1, 2, 3]) {
                        const arr = msg.round_placements[String(pid)] || [0, 0, 0];
                        roundPlacementCounts[pid] = [arr[0] || 0, arr[1] || 0, arr[2] || 0];
                    }
                }
                // Detect round transition BEFORE placement sync
                if (msg.total_rounds !== undefined && msg.total_rounds > lastSeenTotalRounds) {
                    onRoundsChanged(msg.total_rounds - lastSeenTotalRounds);
                    lastSeenTotalRounds = msg.total_rounds;
                }
                // Now sync placements (uses updated baselines)
                if (msg.placements && msg.ranked_games) {
                    syncPlacementsFromServer(msg.ranked_games, msg.placements);
                }
                updateWinRates();
                updatePlacementDisplay();
                // Only push to loss/stats charts when step is numeric (training step), not "Gen X/Y"
                const stepLabel = msg.step != null ? String(msg.step) : (msg.buffer_games != null ? `Gen ${msg.buffer_games}` : '');
                if (stepLabel && lossChart && statsChart && (typeof msg.step === 'number' || Number.isInteger(msg.step))) {
                    const placeholder = {
                        step: msg.step,
                        loss: 0,
                        loss_value: 0,
                        loss_reward: 0,
                        loss_policy: 0,
                        loss_focus: 0,
                        avg_game_length: msg.avg_len != null ? msg.avg_len : 0,
                    };
                    updateLossChart(placeholder);
                    updateStatsChart(placeholder);
                }
                break;

            case 'metrics_history':
                // Bulk load history — fully restore ALL dashboard state
                if (msg.data && msg.data.length > 0) {
                    console.log(`Loaded ${msg.data.length} metrics history entries.`);

                    // 1. Populate loss & stats charts
                    msg.data.forEach(m => {
                        updateLossChart(m);
                        updateStatsChart(m);
                    });
                    lossChart.update();
                    statsChart.update();

                    // 2. Restore win/round stats from last entry
                    const last = msg.data[msg.data.length - 1];
                    if (last.total_games !== undefined) totalGames = last.total_games;
                    if (last.win_counts) winCounts = { ...winCounts, ...last.win_counts };
                    if (last.total_rounds !== undefined) {
                        totalRounds = last.total_rounds;
                        lastSeenTotalRounds = last.total_rounds; // prevent fake trigger
                    }
                    if (last.round_wins) {
                        roundWinCounts = {
                            1: last.round_wins['1'] || 0,
                            2: last.round_wins['2'] || 0,
                            3: last.round_wins['3'] || 0,
                            draw: last.round_wins['draw'] || 0,
                        };
                    }
                    if (last.round_placements) {
                        for (const pid of [1, 2, 3]) {
                            const arr = last.round_placements[String(pid)] || [0, 0, 0];
                            roundPlacementCounts[pid] = [arr[0] || 0, arr[1] || 0, arr[2] || 0];
                        }
                    }
                    updateWinRates();
                    updateSummaryBar(last);

                    // 3. Restore placements & score chart from history
                    eloRatings = { 1: ELO_INITIAL, 2: ELO_INITIAL, 3: ELO_INITIAL };
                    eloHistory.labels.length = 0;
                    eloHistory.red.length = 0;
                    eloHistory.green.length = 0;
                    eloHistory.blue.length = 0;
                    scoreHistory.labels.length = 0;
                    scoreHistory.red.length = 0;
                    scoreHistory.green.length = 0;
                    scoreHistory.blue.length = 0;

                    let prevTotalRoundsH = 0;
                    let prevRankedGamesH = 0;
                    let prevCumPoints = { 1: 0, 2: 0, 3: 0 };
                    let basePointsH = { 1: 0, 2: 0, 3: 0 };
                    let baseRankedH = 0;

                    for (const entry of msg.data) {
                        // Placement & cumulative points
                        if (entry.placements && entry.ranked_games) {
                            rankedGames = entry.ranked_games;
                            for (let pid = 1; pid <= 3; pid++) {
                                const counts = entry.placements[String(pid)] || [0, 0, 0];
                                placementCounts[pid] = [...counts];
                                cumulativePoints[pid] = counts[0] * PLACEMENT_POINTS[0]
                                    + counts[1] * PLACEMENT_POINTS[1]
                                    + counts[2] * PLACEMENT_POINTS[2];
                            }

                            const entryRounds = entry.total_rounds || 0;

                            // Push cumulative score keyed by round
                            const lastLabel = scoreHistory.labels.length > 0 ? scoreHistory.labels[scoreHistory.labels.length - 1] : -1;
                            if (entryRounds === lastLabel) {
                                scoreHistory.red[scoreHistory.red.length - 1] = cumulativePoints[1];
                                scoreHistory.green[scoreHistory.green.length - 1] = cumulativePoints[2];
                                scoreHistory.blue[scoreHistory.blue.length - 1] = cumulativePoints[3];
                            } else if (entryRounds > lastLabel) {
                                // Fill intermediate missing rounds to draw flat horizontal lines instead of point-skip slopes
                                if (lastLabel !== -1 && entryRounds > lastLabel + 1) {
                                    for (let r = lastLabel + 1; r < entryRounds; r++) {
                                        scoreHistory.labels.push(r);
                                        scoreHistory.red.push(prevCumPoints[1]);
                                        scoreHistory.green.push(prevCumPoints[2]);
                                        scoreHistory.blue.push(prevCumPoints[3]);
                                    }
                                }
                                scoreHistory.labels.push(entryRounds);
                                scoreHistory.red.push(cumulativePoints[1]);
                                scoreHistory.green.push(cumulativePoints[2]);
                                scoreHistory.blue.push(cumulativePoints[3]);

                                // Clean up overflow points
                                while (scoreHistory.labels.length > MAX_SCORE_POINTS) {
                                    scoreHistory.labels.shift();
                                    scoreHistory.red.shift();
                                    scoreHistory.green.shift();
                                    scoreHistory.blue.shift();
                                }
                            }

                            if (entryRounds > prevTotalRoundsH) {
                                basePointsH = { 1: prevCumPoints[1], 2: prevCumPoints[2], 3: prevCumPoints[3] };
                                baseRankedH = prevRankedGamesH;
                                prevTotalRoundsH = entryRounds;
                            }

                            prevCumPoints = { 1: cumulativePoints[1], 2: cumulativePoints[2], 3: cumulativePoints[3] };
                            prevRankedGamesH = entry.ranked_games;
                        }

                        // ELO: push league_elo as history point at each round change
                        if (entry.total_rounds && entry.league_elo !== undefined && entry.total_rounds > (eloHistory.labels.length > 0 ? eloHistory.labels[eloHistory.labels.length - 1] : -1)) {
                            eloHistory.labels.push(entry.total_rounds);
                            eloHistory.red.push(entry.league_elo);
                            // Approximate other players from round placements
                            const rp = entry.round_placements;
                            if (rp) {
                                const p2first = (rp['2'] || [0, 0, 0])[0];
                                const p3first = (rp['3'] || [0, 0, 0])[0];
                                const tr = entry.total_rounds || 1;
                                eloHistory.green.push(ELO_INITIAL + (p2first / tr - 0.33) * 400);
                                eloHistory.blue.push(ELO_INITIAL + (p3first / tr - 0.33) * 400);
                            } else {
                                eloHistory.green.push(ELO_INITIAL);
                                eloHistory.blue.push(ELO_INITIAL);
                            }
                            if (eloHistory.labels.length > MAX_ELO_POINTS) {
                                eloHistory.labels.shift();
                                eloHistory.red.shift();
                                eloHistory.green.shift();
                                eloHistory.blue.shift();
                            }
                        }
                    }

                    // Set baselines from restored state
                    roundBaselinePoints = { 1: basePointsH[1], 2: basePointsH[2], 3: basePointsH[3] };
                    roundBaselineRankedGames = baseRankedH;
                    roundPoints = {
                        1: cumulativePoints[1] - roundBaselinePoints[1],
                        2: cumulativePoints[2] - roundBaselinePoints[2],
                        3: cumulativePoints[3] - roundBaselinePoints[3],
                    };
                    roundGameIndex = rankedGames - roundBaselineRankedGames;

                    // Update all displays
                    updatePlacementDisplay();
                    if (scoresChart) {
                        scoresChart.data.labels = [...scoreHistory.labels];
                        scoresChart.data.datasets[0].data = [...scoreHistory.red];
                        scoresChart.data.datasets[1].data = [...scoreHistory.green];
                        scoresChart.data.datasets[2].data = [...scoreHistory.blue];
                        scoresChart.update();
                    }
                    updateEloChart();
                    updateEloBadge();
                }
                break;
        }
    }

    // ---- Rankings & Session Processing ----
    function processRankings(rankings) {
        // rankings: [[pid, placement], ...] where placement is 0-indexed (0=1st, 1=2nd, 2=3rd)
        rankedGames++;

        for (const [pid, placement] of rankings) {
            if (pid >= 1 && pid <= 3 && placement >= 0 && placement <= 2) {
                placementCounts[pid][placement]++;
                cumulativePoints[pid] += PLACEMENT_POINTS[placement];
                roundPoints[pid] += PLACEMENT_POINTS[placement];
            }
        }
        roundGameIndex++;

        // Update placement display and score chart
        updatePlacementDisplay();
        updateScoresChart();
    }

    function syncPlacementsFromServer(serverRankedGames, serverPlacements) {
        /**
         * Overwrite local placement/score state with authoritative data from all actors.
         * Also computes within-round points for score chart.
         */
        const originalIndex = roundGameIndex;
        rankedGames = serverRankedGames;
        for (let pid = 1; pid <= 3; pid++) {
            const counts = serverPlacements[String(pid)] || [0, 0, 0];
            placementCounts[pid] = [...counts];
            cumulativePoints[pid] = counts[0] * PLACEMENT_POINTS[0]
                + counts[1] * PLACEMENT_POINTS[1]
                + counts[2] * PLACEMENT_POINTS[2];
            roundPoints[pid] = cumulativePoints[pid] - roundBaselinePoints[pid];
        }

        // Calculate the authoritative index
        const serverIndex = rankedGames - roundBaselineRankedGames;

        // Only push to the chart if the server index is ahead of our local index
        // or if it's identical (which will just update the existing point via dedup)
        // If the server index is unexpectedly behind, do NOT push backwards points!
        if (serverIndex >= originalIndex) {
            roundGameIndex = serverIndex;
            updatePlacementDisplay();
            updateScoresChart();
        } else {
            updatePlacementDisplay();
        }
    }

    // ---- ELO Rating System ----
    // 3-player ELO: decompose into pairwise matchups.
    // For placements [1st, 2nd, 3rd], we have 3 pairs:
    //   1st beats 2nd, 1st beats 3rd, 2nd beats 3rd.
    // Standard ELO update applied to each pair with K-factor per pair.
    function updateElo(rankings) {
        if (rankings.length < 2) return;

        // Build placement map: pid → placement (0=1st, 1=2nd, 2=3rd)
        const plMap = {};
        for (const [pid, pl] of rankings) {
            plMap[pid] = pl;
        }
        const pids = Object.keys(plMap).map(Number);

        // Accumulate deltas for each player (apply all at once to avoid order bias)
        const deltas = {};
        for (const p of pids) deltas[p] = 0;

        // For every pair (A, B) where A placed higher (lower index) than B
        for (let i = 0; i < pids.length; i++) {
            for (let j = i + 1; j < pids.length; j++) {
                const pidA = pids[i];
                const pidB = pids[j];
                // Determine who placed higher
                let winner, loser;
                if (plMap[pidA] < plMap[pidB]) {
                    winner = pidA; loser = pidB;
                } else if (plMap[pidB] < plMap[pidA]) {
                    winner = pidB; loser = pidA;
                } else {
                    // Same placement (shouldn't happen, but treat as draw)
                    const eA = expectedScore(eloRatings[pidA], eloRatings[pidB]);
                    const eB = 1 - eA;
                    deltas[pidA] += ELO_K * (0.5 - eA);
                    deltas[pidB] += ELO_K * (0.5 - eB);
                    continue;
                }

                const eW = expectedScore(eloRatings[winner], eloRatings[loser]);
                const eL = 1 - eW;
                deltas[winner] += ELO_K * (1 - eW);  // won: score=1
                deltas[loser] += ELO_K * (0 - eL);  // lost: score=0
            }
        }

        // Apply deltas
        for (const p of pids) {
            eloRatings[p] = Math.max(100, eloRatings[p] + deltas[p]); // floor at 100
        }

        // Push to history & update chart
        eloHistory.labels.push(rankedGames);
        eloHistory.red.push(Math.round(eloRatings[1]));
        eloHistory.green.push(Math.round(eloRatings[2]));
        eloHistory.blue.push(Math.round(eloRatings[3]));

        if (eloHistory.labels.length > MAX_ELO_POINTS) {
            eloHistory.labels.shift();
            eloHistory.red.shift();
            eloHistory.green.shift();
            eloHistory.blue.shift();
        }

        updateEloChart();
        updateEloBadge();
    }

    /**
     * Standard Elo expected score: probability A wins against B.
     */
    function expectedScore(ratingA, ratingB) {
        return 1.0 / (1.0 + Math.pow(10, (ratingB - ratingA) / 400.0));
    }

    /**
     * Update ELO from a single round's rankings (Best-of-5 round result).
     * Pushes to eloHistory with totalRounds as x-axis. Used only on round_end.
     */
    function updateEloFromRoundRankings(round_rankings) {
        if (!round_rankings || round_rankings.length < 2) return;

        const plMap = {};
        for (const [pid, pl] of round_rankings) {
            plMap[pid] = pl;
        }
        const pids = Object.keys(plMap).map(Number);
        const deltas = {};
        for (const p of pids) deltas[p] = 0;

        for (let i = 0; i < pids.length; i++) {
            for (let j = i + 1; j < pids.length; j++) {
                const pidA = pids[i];
                const pidB = pids[j];
                let winner, loser;
                if (plMap[pidA] < plMap[pidB]) {
                    winner = pidA; loser = pidB;
                } else if (plMap[pidB] < plMap[pidA]) {
                    winner = pidB; loser = pidA;
                } else {
                    const eA = expectedScore(eloRatings[pidA], eloRatings[pidB]);
                    const eB = 1 - eA;
                    deltas[pidA] += ELO_K * (0.5 - eA);
                    deltas[pidB] += ELO_K * (0.5 - eB);
                    continue;
                }
                const eW = expectedScore(eloRatings[winner], eloRatings[loser]);
                const eL = 1 - eW;
                deltas[winner] += ELO_K * (1 - eW);
                deltas[loser] += ELO_K * (0 - eL);
            }
        }
        for (const p of pids) {
            eloRatings[p] = Math.max(100, eloRatings[p] + deltas[p]);
        }

        const roundX = totalRounds;
        const lastLabel = eloHistory.labels.length > 0 ? eloHistory.labels[eloHistory.labels.length - 1] : -1;
        if (lastLabel === roundX) {
            eloHistory.red[eloHistory.red.length - 1] = Math.round(eloRatings[1]);
            eloHistory.green[eloHistory.green.length - 1] = Math.round(eloRatings[2]);
            eloHistory.blue[eloHistory.blue.length - 1] = Math.round(eloRatings[3]);
        } else {
            eloHistory.labels.push(roundX);
            eloHistory.red.push(Math.round(eloRatings[1]));
            eloHistory.green.push(Math.round(eloRatings[2]));
            eloHistory.blue.push(Math.round(eloRatings[3]));
            if (eloHistory.labels.length > MAX_ELO_POINTS) {
                eloHistory.labels.shift();
                eloHistory.red.shift();
                eloHistory.green.shift();
                eloHistory.blue.shift();
            }
        }
        updateEloChart();
        updateEloBadge();
    }

    /**
     * Derive approximate ELO ratings from cumulative placement data.
     * (Removed: We now rely exclusively on authoritative league_elo broadcasts from the server to prevent desyncs)
     */
    function replayEloFromPlacements(placements, totalRanked) {
        // Obsolete
    }

    function expectedScore(ratingA, ratingB) {
        return 1.0 / (1.0 + Math.pow(10, (ratingB - ratingA) / 400));
    }

    function updateEloChart() {
        if (!eloChart) return;
        eloChart.data.labels = [...eloHistory.labels];
        eloChart.data.datasets[0].data = [...eloHistory.red];
        eloChart.data.datasets[1].data = [...eloHistory.green];
        eloChart.data.datasets[2].data = [...eloHistory.blue];
        eloChart.update('none');
    }

    function updateEloBadge() {
        const r = Math.round(eloRatings[1]);
        const g = Math.round(eloRatings[2]);
        const b = Math.round(eloRatings[3]);
        const badge = document.getElementById('elo-badge');
        if (badge) badge.textContent = `${r} / ${g} / ${b}`;
        const sumEl = document.getElementById('sum-elo');
        if (sumEl) sumEl.textContent = `ELO: ${r}/${g}/${b}`;
    }

    function updatePlacementDisplay() {
        const isRound = placementViewMode === 'round';
        const counts_src = isRound ? roundPlacementCounts : placementCounts;
        const total = isRound ? totalRounds : rankedGames;
        const label = isRound ? `${totalRounds} 回合` : `${rankedGames} 局`;

        const el = document.getElementById('placement-total');
        if (el) el.textContent = label;

        for (let pid = 1; pid <= 3; pid++) {
            const counts = counts_src[pid];
            for (let pl = 0; pl < 3; pl++) {
                const cell = document.getElementById(`plc-${pid}-${pl}`);
                if (cell) cell.textContent = counts[pl];
            }
            // Average points per game/round
            const pts = counts[0] * PLACEMENT_POINTS[0] + counts[1] * PLACEMENT_POINTS[1] + counts[2] * PLACEMENT_POINTS[2];
            const avg = total > 0 ? (pts / total).toFixed(1) : '0.0';
            const avgCell = document.getElementById(`plc-${pid}-avg`);
            if (avgCell) avgCell.textContent = avg;
        }
    }

    /**
     * Called when server reports new completed rounds (totalRounds increased).
     * Clears score chart and pushes ELO update derived from round placements.
     */
    function onRoundsChanged(deltaRounds) {
        // Update baselines for next round period.
        // roundPoints = cumulative - baseline naturally drops to ~0.
        roundBaselinePoints = { 1: cumulativePoints[1], 2: cumulativePoints[2], 3: cumulativePoints[3] };
        roundBaselineRankedGames = rankedGames;
        roundPoints = { 1: 0, 2: 0, 3: 0 };
        roundGameIndex = 0;

        // 2. Derive ELO from round placements
        for (let d = 0; d < deltaRounds; d++) {
            const rankings = [];
            const totalPl = roundPlacementCounts[1][0] + roundPlacementCounts[2][0] + roundPlacementCounts[3][0];
            if (totalPl > 0) {
                const pids = [1, 2, 3].sort((a, b) => roundPlacementCounts[b][0] - roundPlacementCounts[a][0]);
                pids.forEach((pid, i) => rankings.push([pid, i]));
            }
            if (rankings.length >= 2) {
                updateEloFromRoundRankings(rankings);
            }
        }
    }

    function updateScoresChart() {
        const roundNum = totalRounds;
        const lastLabel = scoreHistory.labels.length > 0 ? scoreHistory.labels[scoreHistory.labels.length - 1] : -1;
        // Chart shows overall cumulative points across all rounds
        const redY = cumulativePoints[1];
        const greenY = cumulativePoints[2];
        const blueY = cumulativePoints[3];

        if (roundNum === lastLabel) {
            // Same X-value: update last point in-place (dedup)
            scoreHistory.red[scoreHistory.red.length - 1] = redY;
            scoreHistory.green[scoreHistory.green.length - 1] = greenY;
            scoreHistory.blue[scoreHistory.blue.length - 1] = blueY;
        } else if (roundNum > lastLabel) {
            // New X-value: push new point
            scoreHistory.labels.push(roundNum);
            scoreHistory.red.push(redY);
            scoreHistory.green.push(greenY);
            scoreHistory.blue.push(blueY);

            // Trim to max points
            if (scoreHistory.labels.length > MAX_SCORE_POINTS) {
                scoreHistory.labels.shift();
                scoreHistory.red.shift();
                scoreHistory.green.shift();
                scoreHistory.blue.shift();
            }
        }
        // If roundNum < lastLabel, do nothing (we don't want to draw backwards)

        if (scoresChart) {
            scoresChart.data.labels = [...scoreHistory.labels];
            scoresChart.data.datasets[0].data = [...scoreHistory.red];
            scoresChart.data.datasets[1].data = [...scoreHistory.green];
            scoresChart.data.datasets[2].data = [...scoreHistory.blue];
            scoresChart.update('none');
        }
    }

    function updateSessionDisplay() {
        if (!currentSessionInfo) return;
        const { scores, game_idx, session_length } = currentSessionInfo;
        const infoEl = document.getElementById('session-info');
        if (infoEl) {
            infoEl.textContent = `局 ${game_idx + 1}/${session_length}`;
        }
        const sumEl = document.getElementById('sum-session');
        if (sumEl) {
            const s = scores || {};
            sumEl.textContent = `会话: ${game_idx + 1}/${session_length} [${s[1] || 0}:${s[2] || 0}:${s[3] || 0}]`;
        }
    }

    // ---- Board Rendering ----
    function placeStone(row, col, player) {
        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            board[row][col] = player;
        }
    }

    function renderBoard() {
        // Find bounding box of occupied cells
        let minR = BOARD_SIZE, maxR = 0, minC = BOARD_SIZE, maxC = 0;
        let hasStones = false;
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] !== 0) {
                    hasStones = true;
                    minR = Math.min(minR, r);
                    maxR = Math.max(maxR, r);
                    minC = Math.min(minC, c);
                    maxC = Math.max(maxC, c);
                }
            }
        }

        if (!hasStones) {
            // Empty board - show center region
            minR = 40; maxR = 60; minC = 40; maxC = 60;
        }

        // Add padding
        const pad = 5;
        minR = Math.max(0, minR - pad);
        maxR = Math.min(BOARD_SIZE - 1, maxR + pad);
        minC = Math.max(0, minC - pad);
        maxC = Math.min(BOARD_SIZE - 1, maxC + pad);

        const viewRows = maxR - minR + 1;
        const viewCols = maxC - minC + 1;

        // Calculate cell size to fit in canvas
        const wrapper = document.getElementById('live-board-wrapper');
        const maxW = wrapper.clientWidth - 24;
        const maxH = wrapper.clientHeight - 24;
        const cellSize = Math.max(3, Math.min(
            Math.floor(maxW / viewCols),
            Math.floor(maxH / viewRows),
            20
        ));

        boardCanvas.width = viewCols * cellSize;
        boardCanvas.height = viewRows * cellSize;

        // Background (wood texture)
        boardCtx.fillStyle = '#c19a6b';
        boardCtx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

        // Grid lines
        boardCtx.strokeStyle = 'rgba(0,0,0,0.15)';
        boardCtx.lineWidth = 0.5;
        for (let r = 0; r < viewRows; r++) {
            const y = r * cellSize + cellSize / 2;
            boardCtx.beginPath();
            boardCtx.moveTo(0, y);
            boardCtx.lineTo(viewCols * cellSize, y);
            boardCtx.stroke();
        }
        for (let c = 0; c < viewCols; c++) {
            const x = c * cellSize + cellSize / 2;
            boardCtx.beginPath();
            boardCtx.moveTo(x, 0);
            boardCtx.lineTo(x, viewRows * cellSize);
            boardCtx.stroke();
        }

        // Stones
        const stoneRadius = cellSize * 0.4;
        for (let r = minR; r <= maxR; r++) {
            for (let c = minC; c <= maxC; c++) {
                const p = board[r][c];
                if (p === 0) continue;
                const cx = (c - minC) * cellSize + cellSize / 2;
                const cy = (r - minR) * cellSize + cellSize / 2;

                // Shadow
                boardCtx.beginPath();
                boardCtx.arc(cx + 1, cy + 1, stoneRadius, 0, Math.PI * 2);
                boardCtx.fillStyle = 'rgba(0,0,0,0.25)';
                boardCtx.fill();

                // Stone
                const grad = boardCtx.createRadialGradient(
                    cx - stoneRadius * 0.3, cy - stoneRadius * 0.3, stoneRadius * 0.1,
                    cx, cy, stoneRadius
                );
                grad.addColorStop(0, PLAYER_GLOWS[p] || '#fff');
                grad.addColorStop(1, PLAYER_COLORS[p] || '#888');
                boardCtx.beginPath();
                boardCtx.arc(cx, cy, stoneRadius, 0, Math.PI * 2);
                boardCtx.fillStyle = grad;
                boardCtx.fill();
            }
        }

        // Highlight last move
        if (lastMove) {
            const { row, col } = lastMove;
            const cx = (col - minC) * cellSize + cellSize / 2;
            const cy = (row - minR) * cellSize + cellSize / 2;

            boardCtx.beginPath();
            boardCtx.arc(cx, cy, stoneRadius + 2, 0, Math.PI * 2);
            boardCtx.strokeStyle = 'rgba(255,255,255,0.9)';
            boardCtx.lineWidth = 2.5;
            boardCtx.stroke();
        }
    }

    // ---- Win Rates ----
    function updateWinRates() {
        const isRound = winrateViewMode === 'round';
        const total = isRound ? totalRounds : totalGames;
        const counts = isRound ? roundWinCounts : winCounts;
        const label = isRound ? `${totalRounds} 回合` : `${totalGames} 局`;

        document.getElementById('total-games').textContent = label;
        const rates = total === 0
            ? { red: 0, green: 0, blue: 0, draw: 0 }
            : {
                red: (counts[1] / total) * 100,
                green: (counts[2] / total) * 100,
                blue: (counts[3] / total) * 100,
                draw: ((counts.draw || 0) / total) * 100,
            };

        for (const [key, pct] of Object.entries(rates)) {
            document.getElementById(`wr-${key}`).style.width = pct + '%';
            document.getElementById(`wr-${key}-pct`).textContent = pct.toFixed(1) + '%';
        }
    }

    // ---- Charts ----
    function initCharts() {
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#8a8aaa', font: { size: 9, family: 'JetBrains Mono' }, maxTicksLimit: 8 },
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#8a8aaa', font: { size: 9, family: 'JetBrains Mono' } },
                },
            },
            plugins: {
                legend: {
                    labels: { color: '#ccc', font: { size: 10 }, boxWidth: 10, padding: 6 },
                    position: 'top',
                },
            },
        };

        // Loss chart
        lossChart = new Chart(document.getElementById('loss-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Total', data: [], borderColor: '#a29bfe', borderWidth: 2, pointRadius: 0, tension: 0.3 },
                    { label: 'Value', data: [], borderColor: '#ff6b6b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Reward', data: [], borderColor: '#6bff6b', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Policy', data: [], borderColor: '#6bb5ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: true },
                    { label: 'Focus', data: [], borderColor: '#f1c40f', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: false },
                    { label: 'Recon', data: [], borderColor: '#e84393', borderWidth: 1.5, pointRadius: 0, tension: 0.3, hidden: false },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '训练步数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: { ...commonOptions.scales.y, type: 'logarithmic' },
                },
            },
        });

        // Scores chart (cumulative points per player)
        scoresChart = new Chart(document.getElementById('scores-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '红方', data: [],
                        borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                    {
                        label: '绿方', data: [],
                        borderColor: '#27ae60', backgroundColor: 'rgba(39,174,96,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                    {
                        label: '蓝方', data: [],
                        borderColor: '#2980b9', backgroundColor: 'rgba(41,128,185,0.08)',
                        borderWidth: 2, pointRadius: 0, tension: 0.3, fill: true,
                    },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '回合数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: {
                        ...commonOptions.scales.y,
                        title: { display: true, text: '累计积分', color: '#8a8aaa', font: { size: 9 } },
                    },
                },
            },
        });

        // ELO chart (per-player rating curves)
        eloChart = new Chart(document.getElementById('elo-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '红方', data: [],
                        borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                    {
                        label: '绿方', data: [],
                        borderColor: '#27ae60', backgroundColor: 'rgba(39,174,96,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                    {
                        label: '蓝方', data: [],
                        borderColor: '#2980b9', backgroundColor: 'rgba(41,128,185,0.06)',
                        borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false,
                    },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '回合数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: {
                        ...commonOptions.scales.y,
                        title: { display: true, text: 'ELO', color: '#8a8aaa', font: { size: 9 } },
                    },
                },
                plugins: {
                    ...commonOptions.plugins,
                    // Reference line at 1500
                    annotation: undefined,
                },
            },
        });

        // Stats chart (game length + LR)
        statsChart = new Chart(document.getElementById('stats-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: '游戏长度', data: [], borderColor: '#f39c12', borderWidth: 2, pointRadius: 0, tension: 0.3, yAxisID: 'y' },
                    { label: '红方胜率%', data: [], borderColor: '#e74c3c', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                    { label: '绿方胜率%', data: [], borderColor: '#27ae60', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                    { label: '蓝方胜率%', data: [], borderColor: '#2980b9', borderWidth: 1.5, pointRadius: 0, tension: 0.3, yAxisID: 'y1' },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    x: {
                        ...commonOptions.scales.x,
                        title: { display: true, text: '训练步数', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y: {
                        ...commonOptions.scales.y,
                        position: 'left',
                        title: { display: true, text: '游戏长度', color: '#8a8aaa', font: { size: 9 } },
                    },
                    y1: {
                        ...commonOptions.scales.y,
                        position: 'right',
                        min: 0, max: 100,
                        title: { display: true, text: '胜率%', color: '#8a8aaa', font: { size: 9 } },
                        grid: { drawOnChartArea: false },
                    },
                },
            },
        });
        // Show initial ELO line (1500) when no round data yet
        if (eloHistory.labels.length === 0) {
            eloHistory.labels.push(0);
            eloHistory.red.push(ELO_INITIAL);
            eloHistory.green.push(ELO_INITIAL);
            eloHistory.blue.push(ELO_INITIAL);
            updateEloChart();
        }
    }

    function updateLossChart(metrics) {
        if (typeof metrics.step !== 'number' && !Number.isInteger(metrics.step)) return;
        // Clear stale "Gen X/Y" labels when first numeric step arrives
        if (lossChart.data.labels.some(l => String(l).startsWith('Gen'))) {
            lossChart.data.labels = [];
            lossChart.data.datasets.forEach(d => { d.data = []; });
        }
        const label = String(metrics.step);
        lossChart.data.labels.push(label);
        lossChart.data.datasets[0].data.push(metrics.loss);
        lossChart.data.datasets[1].data.push(metrics.loss_value);
        lossChart.data.datasets[2].data.push(metrics.loss_reward);
        lossChart.data.datasets[3].data.push(metrics.loss_policy);
        lossChart.data.datasets[4].data.push(metrics.loss_focus);
        lossChart.data.datasets[5].data.push(metrics.loss_recon || 0.0);

        // Keep max 200 points
        if (lossChart.data.labels.length > 200) {
            lossChart.data.labels.shift();
            lossChart.data.datasets.forEach(d => d.data.shift());
        }
        lossChart.update('none');

        document.getElementById('current-loss').textContent = `Loss: ${metrics.loss.toFixed(4)} | F: ${metrics.loss_focus.toFixed(4)} | Rec: ${(metrics.loss_recon || 0.0).toFixed(4)}`;
    }

    function updateStatsChart(metrics) {
        if (typeof metrics.step !== 'number' && !Number.isInteger(metrics.step)) return;
        // Clear stale "Gen X/Y" labels when first numeric step arrives
        if (statsChart.data.labels.some(l => String(l).startsWith('Gen'))) {
            statsChart.data.labels = [];
            statsChart.data.datasets.forEach(d => { d.data = []; });
        }
        const label = String(metrics.step);
        statsChart.data.labels.push(label);
        statsChart.data.datasets[0].data.push(metrics.avg_game_length || 0);

        // Cumulative win rates
        const total = totalGames || 1;
        statsChart.data.datasets[1].data.push((winCounts[1] / total) * 100);
        statsChart.data.datasets[2].data.push((winCounts[2] / total) * 100);
        statsChart.data.datasets[3].data.push((winCounts[3] / total) * 100);

        if (statsChart.data.labels.length > 200) {
            statsChart.data.labels.shift();
            statsChart.data.datasets.forEach(d => d.data.shift());
        }
        statsChart.update('none');
    }

    function updateSummaryBar(metrics) {
        if (metrics.step !== undefined)
            document.getElementById('sum-step').textContent = `Step: ${metrics.step}`;
        if (metrics.iteration !== undefined)
            document.getElementById('sum-iter').textContent = `Iteration: ${metrics.iteration}`;
        if (metrics.lr !== undefined)
            document.getElementById('sum-lr').textContent = `LR: ${metrics.lr.toFixed(6)}`;
        if (metrics.buffer_games !== undefined)
            document.getElementById('sum-buf').textContent = `Buffer: ${metrics.buffer_games}`;

        const elapsed = ((Date.now() - startTime) / 60000).toFixed(1);
        document.getElementById('sum-time').textContent = `运行时间: ${elapsed}min`;
    }

    // ---- Start ----
    window.addEventListener('DOMContentLoaded', init);
    // Resize handler
    window.addEventListener('resize', () => renderBoard());
})();
