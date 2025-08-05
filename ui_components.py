# ============================================
# ui_components.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Any, Optional, Dict, List

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> None:
        """Render a styled metric card"""
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        """Render enhanced summary dashboard"""
        from market_intelligence import MarketIntelligence
        
        if df.empty:
            st.warning("No data available for summary")
            return
        
        st.markdown("### 📊 Market Pulse")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            
            if ad_ratio > 2:
                ad_emoji = "🔥"
            elif ad_ratio > 1:
                ad_emoji = "📈"
            else:
                ad_emoji = "📉"
            
            UIComponents.render_metric_card(
                "A/D Ratio",
                f"{ad_emoji} {ad_ratio:.2f}",
                f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}",
                "Advance/Decline Ratio"
            )
        
        with col2:
            high_momentum = len(df[df['momentum_score'] >= 70]) if 'momentum_score' in df.columns else 0
            momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
            
            UIComponents.render_metric_card(
                "Momentum Health",
                f"{momentum_pct:.0f}%",
                f"{high_momentum} strong stocks"
            )
        
        with col3:
            avg_rvol = df['rvol'].median() if 'rvol' in df.columns else 1.0
            high_vol_count = len(df[df['rvol'] > 2]) if 'rvol' in df.columns else 0
            
            if avg_rvol > 1.5:
                vol_emoji = "🌊"
            elif avg_rvol > 1.2:
                vol_emoji = "💧"
            else:
                vol_emoji = "🏜️"
            
            UIComponents.render_metric_card(
                "Volume State",
                f"{vol_emoji} {avg_rvol:.1f}x",
                f"{high_vol_count} surges"
            )
        
        with col4:
            risk_factors = 0
            
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                overextended = len(df[(df['from_high_pct'] >= 0) & (df['momentum_score'] < 50)])
                if overextended > 20:
                    risk_factors += 1
            
            if 'rvol' in df.columns:
                pump_risk = len(df[(df['rvol'] > 10) & (df['master_score'] < 50)])
                if pump_risk > 10:
                    risk_factors += 1
            
            if 'trend_quality' in df.columns:
                downtrends = len(df[df['trend_quality'] < 40])
                if downtrends > len(df) * 0.3:
                    risk_factors += 1
            
            risk_levels = ["🟢 LOW", "🟡 MODERATE", "🟠 HIGH", "🔴 EXTREME"]
            risk_level = risk_levels[min(risk_factors, 3)]
            
            UIComponents.render_metric_card(
                "Risk Level",
                risk_level,
                f"{risk_factors} factors"
            )
        
        st.markdown("### 🎯 Today's Best Opportunities")
        
        opp_col1, opp_col2, opp_col3 = st.columns(3)
        
        with opp_col1:
            if all(col in df.columns for col in ['momentum_score', 'acceleration_score', 'rvol', 'master_score']):
                ready_to_run = df[
                    (df['momentum_score'] >= 70) & 
                    (df['acceleration_score'] >= 70) &
                    (df['rvol'] >= 2)
                ].nlargest(5, 'master_score')
                
                st.markdown("**🚀 Ready to Run**")
                if len(ready_to_run) > 0:
                    for _, stock in ready_to_run.iterrows():
                        st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                        st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {stock['rvol']:.1f}x")
                else:
                    st.info("No momentum leaders found")
        
        with opp_col2:
            if 'patterns' in df.columns and 'master_score' in df.columns:
                hidden_gems = df[df['patterns'].str.contains('HIDDEN GEM', na=False)].nlargest(5, 'master_score')
                
                st.markdown("**💎 Hidden Gems**")
                if len(hidden_gems) > 0:
                    for _, stock in hidden_gems.iterrows():
                        st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                        st.caption(f"Cat %ile: {stock.get('category_percentile', 0):.0f} | Score: {stock['master_score']:.1f}")
                else:
                    st.info("No hidden gems today")
        
        with opp_col3:
            if 'rvol' in df.columns and 'master_score' in df.columns:
                volume_alerts = df[df['rvol'] > 3].nlargest(5, 'master_score')
                
                st.markdown("**⚡ Volume Alerts**")
                if len(volume_alerts) > 0:
                    for _, stock in volume_alerts.iterrows():
                        st.write(f"• **{stock['ticker']}** - {stock['company_name'][:25]}")
                        st.caption(f"RVOL: {stock['rvol']:.1f}x | {stock.get('wave_state', 'N/A')}")
                else:
                    st.info("No extreme volume detected")
        
        st.markdown("### 🧠 Market Intelligence")
        
        intel_col1, intel_col2 = st.columns([2, 1])
        
        with intel_col1:
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            
            if not sector_rotation.empty:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sector_rotation.index[:10],
                    y=sector_rotation['flow_score'][:10],
                    text=[f"{val:.1f}" for val in sector_rotation['flow_score'][:10]],
                    textposition='outside',
                    marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                 for score in sector_rotation['flow_score'][:10]],
                    hovertemplate=(
                        'Sector: %{x}<br>'
                        'Flow Score: %{y:.1f}<br>'
                        'Analyzed: %{customdata[0]} of %{customdata[1]} stocks<br>'
                        'Avg Score: %{customdata[2]:.1f}<br>'
                        'Median Score: %{customdata[3]:.1f}<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        sector_rotation['analyzed_stocks'][:10],
                        sector_rotation['total_stocks'][:10],
                        sector_rotation['avg_score'][:10],
                        sector_rotation['median_score'][:10]
                    ))
                ))
                
                fig.update_layout(
                    title="Sector Rotation Map - Smart Money Flow (Dynamically Sampled)",
                    xaxis_title="Sector",
                    yaxis_title="Flow Score",
                    height=400,
                    template='plotly_white',
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sector rotation data available for visualization.")
        
        with intel_col2:
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            
            st.markdown(f"**🎯 Market Regime**")
            st.markdown(f"### {regime}")
            
            st.markdown("**📡 Key Signals**")
            
            signals = []
            
            breadth = regime_metrics.get('breadth', 0.5)
            if breadth > 0.6:
                signals.append("✅ Strong breadth")
            elif breadth < 0.4:
                signals.append("⚠️ Weak breadth")
            
            category_spread = regime_metrics.get('category_spread', 0)
            if category_spread > 10:
                signals.append("🔄 Small caps leading")
            elif category_spread < -10:
                signals.append("🛡️ Large caps defensive")
            
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 1.5:
                signals.append("🌊 High volume activity")
            
            pattern_count = (df['patterns'] != '').sum()
            if pattern_count > len(df) * 0.2:
                signals.append("🎯 Many patterns emerging")
            
            for signal in signals:
                st.write(signal)
            
            st.markdown("**💪 Market Strength**")
            
            strength_score = (
                (breadth * 50) +
                (min(avg_rvol, 2) * 25) +
                ((pattern_count / len(df)) * 25)
            )
            
            if strength_score > 70:
                strength_meter = "🟢🟢🟢🟢🟢"
            elif strength_score > 50:
                strength_meter = "🟢🟢🟢🟢⚪"
            elif strength_score > 30:
                strength_meter = "🟢🟢🟢⚪⚪"
            else:
                strength_meter = "🟢🟢⚪⚪⚪"
            
            st.write(strength_meter)
