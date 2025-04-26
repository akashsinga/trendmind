def print_top_signals(df, top_n=10):
    if df.empty:
        print("[INFO] No predictions available.")
        return

    # Separate bullish and bearish
    bullish = df[df["prediction"] == "bullish"].sort_values(by="confidence", ascending=False)
    bearish = df[df["prediction"] == "bearish"].sort_values(by="confidence", ascending=False)

    print("\n🟢 [TOP BULLISH PREDICTIONS]")
    print(bullish[["symbol", "confidence"]].head(top_n).to_string(index=False))

    print("\n🔴 [TOP BEARISH PREDICTIONS]")
    print(bearish[["symbol", "confidence"]].head(top_n).to_string(index=False))

    # 📈 Print average confidence
    if not bullish.empty:
        print(f"\n📈 Average Bullish Confidence: {bullish['confidence'].mean():.4f}")
    if not bearish.empty:
        print(f"📉 Average Bearish Confidence: {bearish['confidence'].mean():.4f}")
