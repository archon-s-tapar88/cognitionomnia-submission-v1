with open(r'C:\Users\sahen\cognitionomnia-submission-v1\team_code.py', 'r') as f:
    content = f.read()

old = '''        except Exception as e:
            if verbose:
                tqdm.write(f"  Skip {patient_id}: {e}")
            continue

    pbar.close()'''

new = '''        except Exception as e:
            if verbose:
                tqdm.write(f"  Skip {patient_id}: {e}")
            continue
        
        # Force garbage collection to prevent memory fragmentation
        import gc
        gc.collect()

    pbar.close()'''

content = content.replace(old, new)

with open(r'C:\Users\sahen\cognitionomnia-submission-v1\team_code.py', 'w') as f:
    f.write(content)

print("Added gc.collect() to training loop!")
