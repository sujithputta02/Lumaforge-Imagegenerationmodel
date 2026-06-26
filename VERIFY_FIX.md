# Verifying the API Fix

## Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd /Users/sujithputta/Projects/LumaForge/model
python app.py
```

You should see output like:
```
Starting LumaForge API Server on port 7860...
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
```

## Step 2: Verify API Health

In another terminal, test the health endpoint:

```bash
curl http://127.0.0.1:7860/api/status
```

Expected response:
```json
{
  "status": "healthy",
  "device": "mps",
  "mps_available": true,
  "ollama_connected": true,
  "backend": "FastAPI + PyTorch",
  "timestamp": "2026-06-26T12:34:56Z"
}
```

## Step 3: Test Models Endpoint

```bash
curl http://127.0.0.1:7860/api/models/available
```

Should return available models list.

## Step 4: Test Session Generation

Create a generation session:

```bash
SESSION=$(curl -s -X POST http://127.0.0.1:7860/api/generate-session/start \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "steps": 20,
    "guidance_scale": 7.5,
    "mock": true
  }' | jq -r '.session_id')

echo "Session ID: $SESSION"
```

Check session status:

```bash
curl -X POST http://127.0.0.1:7860/api/generate-session/status \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\"}" | jq
```

You should see status progression:
- `"status": "pending"` → `"status": "running"` → `"status": "completed"`

## Step 5: Start the Frontend

In a third terminal:

```bash
cd /Users/sujithputta/Projects/LumaForge/web
npm run dev
# or
yarn dev
# or
bun dev
```

Frontend runs on `http://localhost:3000`

## Step 6: Test Frontend UI

1. Open `http://localhost:3000` in your browser
2. Go to the **Playground** tab
3. Enter a prompt: "A futuristic city"
4. Click **Generate**
5. You should see:
   - The UI shows "Generation started in background..."
   - No more 404 errors in the browser console
   - The generation completes and displays an image

## Step 7: Check Browser Console

Press `F12` to open developer tools → Console tab

Look for:
- ❌ Red errors about `/api/generate-session/*` (should NOT see these)
- ✅ Normal console logs about generation progress
- ✅ Image loads successfully

## Troubleshooting

### Error: "Connection refused on 127.0.0.1:7860"
- Make sure the backend is running (Step 1)
- Check the port 7860 is not already in use: `lsof -i :7860`

### Error: "Module not found: lumaforge"
- Make sure you're in the `model` directory when running `python app.py`
- The imports use relative paths that expect this working directory

### Error: "Endpoint not found"
- Verify the backend is updated with the latest `app.py`
- Check that `python app.py` started without errors

### Frontend still shows 404s
- Hard refresh the browser: `Cmd+Shift+R`
- Clear browser cache
- Check `BACKEND_URL` is correctly set (defaults to `http://127.0.0.1:7860`)

## Environment Variables (if needed)

Set in your shell before running:

```bash
# Override backend URL
export BACKEND_URL=http://127.0.0.1:7860

# Override port
export PORT=7860

# Run backend
python app.py
```

## Expected Behavior After Fix

✅ **Backend**
- All 28 API endpoints are available
- Session-based generation works
- No 404 errors from frontend requests

✅ **Frontend**
- All tabs load without errors
- Generation shows live progress
- Image processing features work
- No console errors about missing endpoints

✅ **Console Output**
- Browser console: No 404 errors
- Backend console: Logs show successful requests

## Performance Considerations

- First request may be slow (model loading)
- Mock mode (`mock: true`) is instant for testing
- Real generation takes 10-30 seconds depending on steps
- Session cleanup runs every 5 minutes

## Security Notes

- CORS is enabled for all origins (update for production)
- Rate limiting: 10 generations/min, 60 API calls/min per IP
- No authentication (add for production)
- No input validation on image B64 (add for production)

---

**If everything passes these steps, the API fix is working correctly!** 🎉

