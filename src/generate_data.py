#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic mental health posts with labels and timestamps.

This script generates synthetic mental health posts that simulate content from
forums like r/depression, r/anxiety, r/mentalhealth, and mental health journaling apps.
Each post is assigned a binary risk label (0 for low risk, 1 for high risk) and
a random timestamp within the last year.

Output:
    - data/raw/generated_dataset.csv: CSV file containing the generated dataset
      with columns: text, label, timestamp
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta
import argparse

os.makedirs('data/raw', exist_ok=True)


def generate_high_risk_post():
    """Generate a synthetic high-risk mental health post."""
    templates = [
        "I can't take it anymore. Everything feels pointless and I don't see a reason to continue. {detail}",
        "I've been thinking about ending it all. {detail} No one would miss me anyway.",
        "I feel completely hopeless. {detail} I just want the pain to stop.",
        "I'm a burden to everyone around me. {detail} They'd be better off without me.",
        "I've been planning how to kill myself. {detail} I don't see any other way out.",
        "Everything is falling apart and I can't handle it anymore. {detail} I just want to disappear.",
        "I've been cutting myself again. {detail} The physical pain distracts from the emotional torture.",
        "I hate myself so much. {detail} I don't deserve to live.",
        "I've written my goodbye note. {detail} I'm just waiting for the right moment.",
        "The voices won't stop telling me to hurt myself. {detail} I'm afraid I might listen to them."
    ]

    details = [
        "I've been feeling this way for months now.",
        "My depression has never been this bad before.",
        "I can't sleep, I can't eat, I can't function.",
        "I've tried medication but nothing helps.",
        "I'm completely alone in this world.",
        "No one understands what I'm going through.",
        "I've lost my job and have no purpose anymore.",
        "My family would be better off without me.",
        "I've been stockpiling pills just in case.",
        "I feel like I'm drowning and can't breathe.",
        "The darkness is consuming me completely.",
        "I've never felt this empty and hollow before.",
        "I'm exhausted from pretending to be okay.",
        "I've been having vivid thoughts about death.",
        "I don't see any future for myself."
    ]

    template = random.choice(templates)
    detail = random.choice(details)

    return template.format(detail=detail)


def generate_low_risk_post():
    """Generate a synthetic low-risk mental health post."""
    templates = [
        "I've been feeling a bit down lately. {detail} But I'm trying to stay positive.",
        "Work has been stressful this week. {detail} I need to find better coping mechanisms.",
        "I had an anxiety attack yesterday. {detail} Breathing exercises helped me through it.",
        "I'm worried about my upcoming exam. {detail} I'm studying hard but still feel nervous.",
        "My friend and I had an argument. {detail} I hope we can work things out.",
        "I've been feeling unmotivated lately. {detail} Looking for ways to get my energy back.",
        "Sometimes I feel like no one understands me. {detail} But I know that's not really true.",
        "I had trouble sleeping last night. {detail} Going to try meditation tonight.",
        "I'm feeling a bit overwhelmed with responsibilities. {detail} Need to prioritize better.",
        "My mood has been up and down. {detail} I'm trying to identify my triggers."
    ]

    details = [
        "I've been exercising more to help with my mood.",
        "I'm considering talking to a therapist about it.",
        "My friends have been supportive.",
        "I'm trying to maintain a routine to stay grounded.",
        "I've been journaling to process my feelings.",
        "I know this is temporary and will pass.",
        "I'm practicing self-care when I can.",
        "I'm trying to be kinder to myself during this time.",
        "I've been reading self-help books that give me perspective.",
        "I'm focusing on small victories each day.",
        "I've been more open with my family about my feelings.",
        "I'm learning to set better boundaries.",
        "I'm trying new hobbies to distract myself positively.",
        "I know I need to improve my sleep habits.",
        "I'm working on challenging negative thoughts."
    ]

    template = random.choice(templates)
    detail = random.choice(details)

    return template.format(detail=detail)


def generate_ambiguous_post():
    """Generate a synthetic ambiguous mental health post."""
    templates = [
        "Lately I’ve been feeling lost. {detail} I'm not sure what to make of it.",
        "Sometimes things feel okay, sometimes they don’t. {detail}",
        "I don’t know if this is normal or not. {detail}",
        "Some days I feel like giving up, other days I feel hopeful. {detail}",
        "I can’t tell if I’m just tired or something more. {detail}",
        "I keep overthinking everything. {detail} It’s exhausting.",
        "There’s a heaviness I can’t explain. {detail}",
        "I’ve been isolating myself lately. {detail} It’s just easier.",
        "I feel like I’m on autopilot. {detail}",
        "I’ve been struggling to make sense of my feelings. {detail}"
    ]

    details = [
        "But I’m still trying to keep it together.",
        "I haven’t talked to anyone about it yet.",
        "Maybe I just need some rest.",
        "I feel like I’m stuck in a loop.",
        "It’s hard to focus on anything.",
        "I hope it gets better, but I’m not sure.",
        "I’m not sure if I need help or not.",
        "I’m doing what I can, even if it’s not much.",
        "I wish I knew what to do.",
        "I don’t want to worry anyone."
    ]

    template = random.choice(templates)
    detail = random.choice(details)

    return template.format(detail=detail)


def generate_random_timestamp(start_date=None):
    """Generate a random timestamp within the last year."""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)

    end_date = datetime.now()
    random_date = start_date + timedelta(days=random.randint(0, 364))
    random_date = random_date.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )
    return random_date.strftime('%Y-%m-%d %H:%M:%S')


def generate_dataset(num_samples=800):
    """Generate a synthetic dataset with high risk, low risk, and ambiguous posts."""
    data = []

    num_high_risk = int(num_samples * 0.3)
    num_ambiguous = int(num_samples * 0.3)
    num_low_risk = num_samples - num_high_risk - num_ambiguous

    for _ in range(num_high_risk):
        text = generate_high_risk_post()
        label = 1
        timestamp = generate_random_timestamp()
        data.append({'text': text, 'label': label, 'timestamp': timestamp})

    for _ in range(num_low_risk):
        text = generate_low_risk_post()
        label = 0
        timestamp = generate_random_timestamp()
        data.append({'text': text, 'label': label, 'timestamp': timestamp})

    for _ in range(num_ambiguous):
        text = generate_ambiguous_post()
        label = random.choice([0, 1])  # Assign ambiguous posts randomly
        timestamp = generate_random_timestamp()
        data.append({'text': text, 'label': label, 'timestamp': timestamp})

    random.shuffle(data)
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic mental health posts dataset')
    parser.add_argument('--num_samples', type=int, default=800, help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='data/raw/generated_dataset.csv', help='Output file path')
    args = parser.parse_args()

    print(f"Generating {args.num_samples} synthetic mental health posts...")
    df = generate_dataset(num_samples=args.num_samples)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output}")
    high_risk_count = df['label'].sum()
    low_risk_count = len(df) - high_risk_count
    print(f"Total samples: {len(df)}")
    print(f"High risk samples: {high_risk_count} ({(high_risk_count / len(df)) * 100:.1f}%)")
    print(f"Low risk samples: {low_risk_count} ({(low_risk_count / len(df)) * 100:.1f}%)")


if __name__ == "__main__":
    main()
