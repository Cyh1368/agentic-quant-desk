# darkzodchi on X: "How to run a quant desk with four AI agents and zero people "

*by darkzodchi (@zodchiii)*

> Source: https://x.com/zodchiii/status/2075874969930662160

![Image](https://pbs.twimg.com/media/HM7qoeqWQAMGx5V?format=jpg&name=orig)

You keep trading by hand, getting chopped up, and telling yourself you'll build a bot one day. Then you remember you can't code.

This is how I built one anyway: the research, the rules in plain English, and turning it into a bot that trades itself.

No code, no all-nighters staring at charts. You describe it once and it runs.

Here's the whole build you need 👇

Before we dive in, I share daily notes on AI & vibe coding in my Telegram channel: [https://t.me/zodchixquant](https://t.me/zodchixquant) 🧠

![Image](https://pbs.twimg.com/media/HM7pciPWEAA96kK?format=jpg&name=orig)

## The desk, and the one thing we are building

A trading desk is four jobs: an analyst who reads the market, a quant who writes strategies, an execution trader who places orders, and a risk manager who stops the bleeding.

Minara runs all four as agents on one account.

![Image](https://pbs.twimg.com/media/HM7q4nJWQAAZOzK?format=jpg&name=orig)

Instead of touring all four, I made them build one real thing: a SOL trend-following bot on the 1-hour that goes long and short, from the first hunch to a live position.

Every screenshot is a step I actually did👇

## 1. The hunch

SOL trends for weeks, then chops sideways for weeks. Trading every move by hand kept getting me chopped up.

I wanted a bot that only trades a clear trend and sits out the mess, so I checked the idea before building it.

```
Over the last 6 months on the 1-hour, how much of the time was SOL in a clean trend versus chopping sideways? Would a trend-follower that can go long or short have beaten buy-and-hold?
```

![Image](https://pbs.twimg.com/media/HM7roObWEAAzX85?format=jpg&name=orig)

This is the part that used to cost me five open tabs and an hour:

![Image](https://pbs.twimg.com/media/HM7rtf3WIAATqM0?format=jpg&name=orig)

The read was clear: SOL had a couple of real trend windows and a lot of chop, and it moved enough in both directions that a long-and-short trend-follower had an edge.

Good enough to turn into rules.

## 2. From a read to exact rules

A hunch cannot be traded by a machine. A rule can. So I wrote the setup down with no room for interpretation, and called it the SOL Regime Trend Follower.

The whole idea is one line: only trade when SOL is genuinely trending, and stay flat the rest of the time. Most bots lose money because they trade constantly.

This one is built to wait.

## 3. Build it in Studio

Here is the vibe coding part.

I did not write a single line. I described the strategy in plain words and Strategy Studio wrote the whole thing in Pine Runtime, Minara's language for strategies.

`Build me a SOL strategy on the 1-hour chart. Only trade when SOL is clearly trending, and stay out when it's moving sideways. Buy when the trend is up, short when the trend is down. Wait for a small pullback before entering instead of chasing. Cut losses quickly with a stop, and get out when the trend flips. Trade smaller when the price is jumpy, bigger when it's calm. After a loss, wait a bit before trying again.
`

```
Build me a SOL strategy on the 1-hour chart. Only trade when SOL is clearly trending, and stay out when it's moving sideways. Buy when the trend is up, short when the trend is down. Wait for a small pullback before entering instead of chasing. Cut losses quickly with a stop, and get out when the trend flips. Trade smaller when the price is jumpy, bigger when it's calm. After a loss, wait a bit before trying again.
```

![Image](https://pbs.twimg.com/media/HM7sHC0WYAAJLB4?format=jpg&name=orig)

My paragraph became working code I could read and edit. You never touch the code unless you want to.

## 4. The first backtest

I backtested the first version on SOL, 1-hour, over six months.

![Image](https://pbs.twimg.com/media/HM7sVtBWMAANph2?format=jpg&name=orig)

It made money, but it lost too often and the drawdown was deeper than I wanted to sit through. On a hand-built platform this is where I would burn a weekend nudging parameters by hand, one run at a time.

I did something faster.

## 5. Push it until the numbers were good

I did not get here from one lucky message. I told Minara what "good" looked like, then went back and forth with it until the strategy actually cleared the bar.

`Improve the strategy until it's solidly profitable and beats buy-and-hold by a wide margin.
Do not stop until it does.
Profit > 30%.`

```
Improve the strategy until it's solidly profitable and beats buy-and-hold by a wide margin. 
Do not stop until it does. 
Profit > 30%.
```

Every round looked the same. Minara ran its Optimize sweep across the moving-average lengths, the trend filter, the stop distance, the cooldown, and the position size, then came back with a new set of parameters and a fresh backtest.

I read the result, kept what improved, told it what still bothered me, and sent it back in.

![Image](https://pbs.twimg.com/media/HM7sdDoXcAA-8PH?format=jpg&name=orig)

It reworked the strategy on its own and came back far stronger: +161% return, beating buy-and-hold SOL by 204% while SOL itself fell 43%. Sharpe 2.10, profit factor 1.37, across 71 trades.

That is the part a hand-built bot cannot do. You state the target, and the assistant optimizes toward it instead of you grinding through fifty versions.

![Image](https://pbs.twimg.com/media/HM7sgt5XIAAl-tp?format=jpg&name=orig)

## 6. Paper trade before real money

A backtest is a promise the past makes. Paper trading checks if it holds right now. I let it run on live prices with no money at stake.

![Image](https://pbs.twimg.com/media/HM7s9E6X0AA_f97?format=jpg&name=orig)

If it fell apart here, it was overfit and I just saved myself the loss. It traded the way the backtest did, so I moved to a tiny live size.

## 7. Go live on Autopilot

When I was satisfied, I sent it to Autopilot. It now trades the rules 24/7, puts a stop on every position, and exits when the trend breaks.

![Image](https://pbs.twimg.com/media/HM7tFBbXsAEwvJb?format=jpg&name=orig)

I set the guardrails before starting: $100 minimum, and a forced exit if equity hits $5. It runs leveraged perps, so this is not free money, and I sized it like I could lose it.

## 8. Keep it honest with a schedule

The last piece turns a one-time build into something that maintains itself. One Workflow checks the setup every morning and messages me if conditions change.

```
Every morning at 8am, check whether SOL is still trending on the 1-hour, and DM me on Telegram if the trend breaks.
```

![Image](https://pbs.twimg.com/media/HM7tL1JXgAAYnYb?format=jpg&name=orig)

Now I am not babysitting a chart. The bot trades, and it tells me when something has shifted.

## The part that lets it hold your keys

Most people never check the one piece that makes it safe to let software touch a wallet.

Minara runs on its own finance model, DMind, split across three places:

![Image](https://pbs.twimg.com/media/HM7tkyLWEAA0INA?format=jpg&name=orig)

A model on your device checks every transaction and keeps your portfolio local. The cloud model only sees market-wide questions and can never sign anything.

Your wallet is an on-chain contract you can audit, not a balance hidden on a server.

Hand-built setups skip this guardrail. DMind is open source, so you can verify the claim instead of trusting it.

## The honest part

None of this removes market risk. Mandatory stops help, but leveraged trades can still liquidate and a clean backtest can fall apart live. Risk only what you can lose, paper trade first, and treat the assistant as a fast analyst, not an oracle.

The strategy above is a build I ran, not a guarantee. Yours will have different numbers. What travels is the process: hunch, research, rules, backtest, improve, live.

Partners include Circle, Ripple, Hyperliquid, Particle Network, Kite, and imToken, with a SlowMist audit on the site.

## Bottom line

A real desk needs four teams. You just ran all four from one chat box: asked what was moving, turned it into a plan, built the strategy, sent it live, and put it on a schedule.

Start with one question and one scheduled workflow. Grow into the full desk when it earns your trust.

- Try it:[ https://minara.ai](https://minara.ai)
- Read the architecture:[ https://minara.ai/docs](https://minara.ai/docs)

![Image](https://pbs.twimg.com/media/HM7ttZgXIAAbmv4?format=jpg&name=orig)
