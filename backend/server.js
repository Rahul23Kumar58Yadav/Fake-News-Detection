// server.js
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 5000;

// HARD-CODED CORRECT URL
const FLASK_ML_API = 'http://localhost:5001/api/predict';
const FLASK_HEALTH_API = 'http://localhost:5001/api/health';
const FLASK_HISTORY_API = 'http://localhost:5001/api/history';
const FLASK_STATS_API = 'http://localhost:5001/api/stats';

console.log('FLASK URL:', FLASK_ML_API); // ← MUST SHOW /api/predict

app.get('/api/health', async(req, res) => {
    try {
        const r = await axios.get(FLASK_HEALTH_API);
        res.json({ status: 'OK', ml: r.data.status });
    } catch {
        res.json({ status: 'OK', ml: 'DOWN' });
    }
});

app.post('/api/predict', async(req, res) => {
    const { text } = req.body;
    if (!text || text.trim().length < 20) {
        return res.status(400).json({ success: false, error: 'Text too short' });
    }

    try {
        console.log('→ Calling Flask:', FLASK_ML_API);
        const r = await axios.post(FLASK_ML_API, { text });
        console.log('← Flask success:', r.data.success);
        res.json(r.data);
    } catch (e) {
        const status = e.response && e.response.status;
        const errorMsg = e.response && e.response.data && e.response.data.error;
        console.error('FLASK ERROR:', status, errorMsg || e.message);
        res.status(502).json({ success: false, error: 'ML failed' });
    }
});
app.get('/api/history', async(req, res) => {
    try {
        const r = await axios.get(FLASK_HISTORY_API, { params: req.query });
        res.json(r.data);
    } catch {
        res.json({ success: true, data: [] });
    }
});

app.get('/api/stats', async(req, res) => {
    try {
        const r = await axios.get(FLASK_STATS_API);
        res.json(r.data);
    } catch {
        res.json({ success: true, stats: { total: 0 } });
    }
});

app.listen(PORT, () => {
    console.log(`Node.js: http://localhost:${PORT}`);
    console.log(`→ Flask: ${FLASK_ML_API}`);
});