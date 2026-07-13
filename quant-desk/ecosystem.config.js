// PM2 config for the quant-desk research jobs (Lightsail host).
// 4h sentiment cadence aligned to the desk's UTC decision times.
module.exports = {
  apps: [
    {
      name: "qd-sentiment-cycle",
      script: "scripts/run_sentiment_cycle.sh",
      cwd: __dirname,
      interpreter: "bash",
      autorestart: false,           // one-shot per cron tick, not a daemon
      cron_restart: "5 0,4,8,12,16,20 * * *",  // :05 past each decision hour UTC
      out_file: "logs/sentiment-cycle.log",
      error_file: "logs/sentiment-cycle.err",
      time: true,
    },
    {
      name: "qd-dashboard",
      script: ".venv/bin/python",
      args: "-m quantdesk.dashboard",
      cwd: __dirname,
      autorestart: true,
      env: { QD_DASHBOARD_PORT: "8420", QD_DASHBOARD_HOST: "127.0.0.1" },
      out_file: "logs/dashboard.log",
      error_file: "logs/dashboard.err",
      time: true,
    },
  ],
};
